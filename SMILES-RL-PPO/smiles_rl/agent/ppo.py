from ..utils.general import to_tensor
import torch
import numpy as np
import os
from ..model.critic_model import CriticModel
from ..model.actor_model import ActorModel

from .base_agent import BaseAgent

from .utils.rewards import rewards_to_go

from typing import Tuple, List

from .utils.sample import sample_unique_sequences

from ..configuration_envelope import ConfigurationEnvelope

from ..logging.training_tracker import TrainingTracker

from copy import deepcopy

import time

from reinvent_scoring import FinalSummary

from rdkit import Chem

from reinvent_chemistry.logging import fraction_valid_smiles


def is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES string is chemically valid using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


class PPO(BaseAgent):
    def __init__(
        self,
        config: ConfigurationEnvelope,
        scoring_function,
        diversity_filter,
        replay_buffer,
        logger,
    ) -> None:
        self._config = config

        self._logger = logger

        self.config = config.reinforcement_learning.parameters

        self.discount_factor = self.config.specific_parameters.get(
            "discount_factor", 0.99
        )

        self.n_updates_per_iteration = self.config.specific_parameters.get(
            "n_updates_per_iteration", 5
        )

        self.clip = self.config.specific_parameters.get("clip", 0.2)

        self.use_entropy_bonus = self.config.specific_parameters.get(
            "use_entropy_bonus", False
        )

        self.entropy_coeff = self.config.specific_parameters.get("entropy_coeff", 0.001)

        self.clip_grad_norm = self.config.specific_parameters.get(
            "clip_grad_norm", False
        )

        self.max_grad_norm = self.config.specific_parameters.get("max_grad_norm", 0.5)

        self.n_minibatches = self.config.specific_parameters.get("n_minibatches", 4)

        # KL divergence penalty to prevent catastrophic forgetting
        # This keeps the agent close to the prior (pre-trained model)
        self.use_kl_penalty = self.config.specific_parameters.get("use_kl_penalty", True)
        self.kl_coeff = self.config.specific_parameters.get("kl_coeff", 0.1)
        self.kl_target = self.config.specific_parameters.get("kl_target", 0.01)  # Target KL for adaptive coefficient
        self.kl_coeff_max = self.config.specific_parameters.get("kl_coeff_max", 1.0)  # Max KL coefficient
        self.adaptive_kl = self.config.specific_parameters.get("adaptive_kl", True)  # Whether to adapt KL coefficient

        # Fine-tuning control
        # If freeze_backbone is True, only the head (output layer) will be trained
        self.freeze_backbone = self.config.specific_parameters.get("freeze_backbone", False)
        
        # Learning rate settings for transformer fine-tuning
        # Use separate learning rates for backbone (pretrained) and head (task-specific)
        self.learning_rate_backbone = self.config.specific_parameters.get(
            "learning_rate_backbone", self.config.learning_rate * 0.1  # Default: 10x smaller than head
        )
        self.learning_rate_head = self.config.specific_parameters.get(
            "learning_rate_head", self.config.learning_rate
        )
        
        # If backbone is frozen, set its learning rate to 0
        if self.freeze_backbone:
            self.learning_rate_backbone = 0.0

        # Reset mechanism parameters (from paper Section 3.2)
        # If validity drops below threshold for patience episodes, reset to prior
        self.reset_validity_threshold = self.config.specific_parameters.get(
            "reset_validity_threshold", 0.8
        )
        self.reset_patience = self.config.specific_parameters.get(
            "reset_patience", 10
        )

        # Invalid score penalty - penalize invalid SMILES to prevent reward hacking
        # Default 0.0 means no penalty (invalid SMILES get score 0)
        # Set to -1.0 to strongly penalize invalid molecules
        self.invalid_score = self.config.specific_parameters.get(
            "invalid_score", 0.0
        )
        
        # Reward scaling factor - amplify reward signal for better gradient updates
        # Default 10.0 helps when raw scores are very small (e.g., 0.01-0.02)
        self.reward_scale = self.config.specific_parameters.get(
            "reward_scale", 1.0
        )

        # Initialize model parameters
        self.reset()

        self._diversity_filter = diversity_filter

        self._replay_buffer = replay_buffer

        self._scoring_function = scoring_function

        self.step = 0

        self.n_invalid_steps = 0
        
        # Track invalid penalty for logging
        self._last_n_invalid = 0
        self._last_invalid_penalty = -10.0
        self._last_min_valid_reward = 0.0
        self._last_sample_n_invalid = 0
        self._last_adjusted_score = None
        self._current_invalid_mask = None
        self._current_smiles = None

        self.start_time = time.time()
        
        # Initialize training tracker for metrics logging and visualization
        run_name = self._config.logging.parameters.get("job_id", "ppo_run")
        log_dir = os.path.dirname(self._config.logging.parameters.get("result_folder", "logs/results"))
        plot_frequency = self._config.logging.parameters.get("plot_frequency", 0)
        self.training_tracker = TrainingTracker(base_log_dir=log_dir, run_name=run_name, plot_frequency=plot_frequency)
        
        # Track historical maximum score across all steps
        self._historical_max_prob = 0.0

    def reset(
        self,
    ) -> None:
        """Reset models"""

        self._critic = CriticModel.load_from_file(
            file_path=self.config.prior, sampling_mode=False
        )
        
        # Freeze backbone if configured (only train the head/output layer)
        if self.freeze_backbone:
            self._critic.freeze_backbone()
        
        # Use separate learning rates for backbone and head (transformer) or single rate (RNN)
        critic_param_groups = self._critic.get_parameter_groups(
            lr_backbone=self.learning_rate_backbone,
            lr_head=self.learning_rate_head,
        )
        self._critic_optimizer = torch.optim.Adam(critic_param_groups)

        self._actor = ActorModel.load_from_file(
            file_path=self.config.agent, sampling_mode=False
        )
        
        # Freeze backbone if configured (only train the head/output layer)
        if self.freeze_backbone:
            self._actor.freeze_backbone()
        
        # Use separate learning rates for backbone and head (transformer) or single rate (RNN)
        actor_param_groups = self._actor.get_parameter_groups(
            lr_backbone=self.learning_rate_backbone,
            lr_head=self.learning_rate_head,
        )
        self._actor_optimizer = torch.optim.Adam(actor_param_groups)

        # Load frozen Prior model for KL divergence regularization
        # This prevents catastrophic forgetting by keeping the agent close to the prior
        if self.use_kl_penalty:
            self._prior = ActorModel.load_from_file(
                file_path=self.config.prior, sampling_mode=True  # inference mode, frozen
            )
            # Freeze all prior parameters
            for param in self._prior.get_network_parameters():
                param.requires_grad = False

        assert (
            self._actor.get_vocabulary() == self._critic.get_vocabulary()
        ), "The agent and the prior must have the same vocabulary"

    @torch.no_grad()
    def act(self, batch_size: int) -> List[str]:
        """Generate sequences and return corresponding SMILES strings

        Args:
            batch_size (int): number of (non-unqiue) sequences to generate

        Returns:
            List[str]: SMILES strings corresponding to unique sequences.
        """
        # Switch to eval mode for sampling (disable dropout)
        self._actor.set_mode("inference")

        self.seqs, self.smiles, self.batch_log_probs = sample_unique_sequences(
            self._actor, batch_size
        )

        # Switch back to training mode for updates
        self._actor.set_mode("training")

        return deepcopy(self.smiles)

    def update(self, smiles: List[str]):
        """Updates policy (actor) using PPO clipped loss and critic using MSE loss

        Args:
            smiles (List[str]): SMILES strings to use for upgrade
        """

        assert (
            self._critic.get_vocabulary() == self._actor.get_vocabulary()
        ), "The actor and the critic must have the same vocabulary"

        try:
            score_summary = self._scoring_function.get_final_score_for_step(
                smiles, self.step
            )
        except TypeError as inst:
            print(inst, flush=True)
            score_summary = FinalSummary(
                np.zeros((len(smiles),), dtype=np.float32), smiles, [], []
            )

        # Score summary includes both valid smiles and invalid smiles
        # Invalid smiles are given scores of 0
        score_summary = deepcopy(score_summary)
        score = self._diversity_filter.update_score(score_summary, self.step)

        # Mark invalid molecules - they will get a FIXED penalty (not scaled)
        # This ensures:
        # 1. Invalid molecules are always worse than valid ones with DRD2 > threshold
        # 2. The penalty doesn't scale with batch composition (stable training)
        # 3. High-scoring batches aren't penalized for having some invalids
        invalid_mask = np.array([not is_valid_smiles(s) for s in smiles], dtype=bool)
        n_invalid = invalid_mask.sum()
        
        # Set invalid molecules to 0 for now - actual penalty applied after scaling
        # This preserves the raw DRD2 scores for valid molecules
        score[invalid_mask] = 0
        self._last_n_invalid = n_invalid
        
        # Store invalid mask for use in reward scaling
        self._current_invalid_mask = invalid_mask.copy()
        self._current_smiles = smiles.copy() if hasattr(smiles, 'copy') else list(smiles)

        # Raw activity probability (after diversity filter, before reward scaling)
        prob_score = score.copy()

        # Extract individual component scores for logging
        # Use valid_mask to compute mean of VALID molecules only
        component_scores = {}  # Mean scores for summary (valid molecules only)
        component_scores_array = {}  # Per-molecule scores for sample display
        valid_mask = ~invalid_mask
        n_valid = valid_mask.sum()
        
        if hasattr(score_summary, 'scaffold_log') and score_summary.scaffold_log:
            for component in score_summary.scaffold_log:
                comp_name = component.parameters.name
                comp_scores = component.total_score
                # Calculate mean for VALID molecules only
                if n_valid > 0:
                    valid_comp_scores = comp_scores[valid_mask]
                    component_scores[comp_name] = np.mean(valid_comp_scores)
                else:
                    component_scores[comp_name] = 0.0
                component_scores_array[comp_name] = comp_scores  # Store full array for distribution stats

        sample_smiles, sample_score = self._replay_buffer(
            score_summary.scored_smiles, score
        )

        sample_seqs = self._actor.smiles_to_sequences(sample_smiles)

        # Calculate policy entropy and KL divergence based on the ACTUAL training batch (after replay buffer)
        with torch.no_grad():
            # Create mask for padding tokens
            pad_token_id = self._actor.vocabulary["$"]
            is_pad = (sample_seqs[:, 1:] == pad_token_id)
            mask = (~is_pad) | (is_pad.cumsum(dim=1) == 1)
            mask = mask.float()

            log_probs, probs = self._actor.log_and_probabilities(sample_seqs)
            entropy = -(probs * log_probs).sum(dim=2)
            policy_entropy = (entropy * mask).sum() / (mask.sum() + 1e-8)
            policy_entropy = policy_entropy.item()

            # Calculate KL divergence for monitoring
            if self.use_kl_penalty:
                prior_log_probs = self._prior.log_probabilities_action(sample_seqs)
                agent_log_probs = self._actor.log_probabilities_action(sample_seqs)
                kl_divergence = ((agent_log_probs - prior_log_probs) * mask).sum() / (mask.sum() + 1e-8)
                kl_divergence = kl_divergence.item()
            else:
                kl_divergence = 0.0
        
        # Apply reward scaling to amplify gradient signal
        # Re-compute invalid mask for sampled molecules (replay buffer might select different subset)
        sample_invalid_mask = np.array([not is_valid_smiles(s) for s in sample_smiles], dtype=bool)
        sample_valid_mask = ~sample_invalid_mask
        
        # Start with scaled scores for all molecules
        adjusted_score = sample_score * self.reward_scale
        
        # RELATIVE INVALID PENALTY: Set invalid to be BELOW the minimum valid score
        # This GUARANTEES invalid molecules are always worse than ANY valid molecule,
        # regardless of how high the scores get!
        # 
        # Strategy: invalid_penalty = min(valid_scores) - MARGIN
        # But also ensure it's always negative (never rewarding invalid molecules)
        PENALTY_MARGIN = 10.0  # Invalid is always at least 10 points below worst valid
        
        valid_rewards = adjusted_score[sample_valid_mask]
        if len(valid_rewards) > 0:
            min_valid_reward = np.min(valid_rewards)
            # Invalid penalty = minimum valid reward - margin
            # Also ensure it's always negative (never positive reward for invalid!)
            invalid_penalty = min(min_valid_reward - PENALTY_MARGIN, -1.0)
        else:
            # Fallback if no valid molecules (shouldn't happen normally)
            invalid_penalty = -PENALTY_MARGIN
        
        adjusted_score[sample_invalid_mask] = invalid_penalty
        
        # Track for logging
        self._last_invalid_penalty = invalid_penalty
        self._last_min_valid_reward = min_valid_reward if len(valid_rewards) > 0 else 0.0
        self._last_sample_n_invalid = sample_invalid_mask.sum()
        
        # Store adjusted scores for logging the actual training reward
        self._last_adjusted_score = adjusted_score.copy()

        # Max score for current (non-replay) batch only
        n_new = len(score_summary.scored_smiles)
        max_current_prob = (
            float(np.max(prob_score[:n_new])) if n_new > 0 else 0.0
        )

        # Calculate rewards for each action
        # Get end token from vocabulary (default 0 for RNN, typically 2 for transformers)
        try:
            end_token = self._actor.vocabulary["$"]
        except (KeyError, TypeError):
            end_token = 0  # Fallback for RNN models
        batch_rtgs = rewards_to_go(
            sample_seqs, to_tensor(adjusted_score), self.discount_factor, end_token=end_token
        )  # [ batch size, number of timesteps]
        # calculate advantages at current step (iteration)
        # Advantages should be detached since gradient should not be over these
        with torch.no_grad():
            _, old_log_probs = self._evaluate(
                sample_seqs
            )  # [batch_size, seqs lenght -1]

        n_batch = sample_seqs.size(0)

        assert n_batch > 0, "Have not sampled any sequences"

        # If we have only sampled non-unique molecules (can easily happen without diversity filter), then
        # it is possible to have less molecules than the minibatch size.
        n_batch_train = (
            n_batch // self.n_minibatches if n_batch > self.n_minibatches else n_batch
        )

        # Compute prior log probs once for KL calculation (if using KL penalty)
        prior_log_probs_all = None
        if self.use_kl_penalty:
            with torch.no_grad():
                prior_log_probs_all = self._prior.log_probabilities_action(sample_seqs)

        # Update network for some n epochs
        kl_divergences = []
        early_stop = False
        
        for epoch in range(self.n_updates_per_iteration):
            if early_stop:
                break
                
            # Calculate V_phi and pi_theta(a_t | s_t)

            actor_loss_mini = []
            critic_loss_mini = []
            kl_mini = []

            with torch.no_grad():
                adv_values, _ = self._evaluate(
                    sample_seqs
                )  # [batch_size, seqs lenght -1]

            adv = batch_rtgs - adv_values

            permut = torch.randperm(n_batch)

            for start in range(0, n_batch, n_batch_train):

                end = start + n_batch_train

                mbinds = permut[start:end]

                # Minibatch values
                mini_seqs = sample_seqs[mbinds, :]

                mini_rewards = batch_rtgs[mbinds, :]

                mini_old_log_probs = old_log_probs[mbinds, :]

                mini_adv = adv[mbinds, :]

                # Normalize advantages per minibatch to increase stability
                mini_adv = (mini_adv - mini_adv.mean()) / (mini_adv.std() + 1e-8)

                values, curr_log_probs = self._evaluate(mini_seqs)

                # Clamp ratios to prevent explosion
                log_ratio = curr_log_probs - mini_old_log_probs
                log_ratio = log_ratio.clamp(-10, 10)  # Prevent exp overflow
                ratios = torch.exp(log_ratio).nan_to_num(nan=1.0, posinf=10.0, neginf=0.1)

                assert mini_adv.size() == ratios.size()

                # Calculate surrogate losses
                surr1 = ratios * mini_adv

                assert surr1.size() == ratios.size()

                clip_range = self.clip

                surr2 = mini_adv * torch.clamp(ratios, 1 - clip_range, 1 + clip_range)

                # Calculate actor and critic losses
                # Use mean over valid tokens only
                pad_token_id = self._actor.vocabulary["$"]
                is_pad = (mini_seqs[:, 1:] == pad_token_id)
                # Keep valid tokens and the first end token (stop action)
                mask = (~is_pad) | (is_pad.cumsum(dim=1) == 1)
                mask = mask.float()
                
                # Masked mean for actor loss
                surrogate_loss = -torch.minimum(surr1, surr2)
                actor_loss = (surrogate_loss * mask).sum() / (mask.sum() + 1e-8)

                actor_loss_mini.append(actor_loss.item())

                if self.use_entropy_bonus:
                    log_probs, probs = self._actor.log_and_probabilities(mini_seqs)
                    # entropy = sum(p * log(p)) which is -H
                    entropy = (probs * log_probs).sum(dim=2)
                    entropy = (entropy * mask).sum() / (mask.sum() + 1e-8)

                    actor_loss += self.entropy_coeff * entropy

                # KL divergence penalty - apply directly to loss for immediate feedback
                # This is critical for preventing catastrophic forgetting
                if self.use_kl_penalty and prior_log_probs_all is not None:
                    mini_prior_log_probs = prior_log_probs_all[mbinds, :]
                    
                    # Compute standard KL divergence: KL(agent || prior) = E[log(agent) - log(prior)]
                    # This is the proper KL formulation used in PPO papers
                    log_ratio_kl = curr_log_probs - mini_prior_log_probs
                    # Clamp more aggressively to prevent numerical explosion
                    log_ratio_kl = log_ratio_kl.clamp(-5, 5)
                    
                    # Mean KL over valid tokens (standard KL, not Schulman's approximation)
                    # This gives reasonable values in range [0, ~5] instead of millions
                    batch_kl = (log_ratio_kl * mask).sum() / (mask.sum() + 1e-8)
                    # Take absolute value since KL should be non-negative
                    batch_kl = batch_kl.abs()
                    kl_mini.append(batch_kl.item())
                    
                    # Add KL penalty to actor loss
                    actor_loss += self.kl_coeff * batch_kl

                critic_loss = 0.5 * torch.nn.functional.mse_loss(values, mini_rewards)

                critic_loss_mini.append(critic_loss.item())

                # Calculate gradient and perform backward propagation for actor network
                self._actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                
                # Clip gradients (always clip for stability with KL penalty)
                torch.nn.utils.clip_grad_norm_(
                    self._actor.get_network_parameters(),
                    self.max_grad_norm,
                    error_if_nonfinite=True,
                )

                self._actor_optimizer.step()

                # Calculate gradients and perform backward propagation for critic network
                self._critic_optimizer.zero_grad()
                critic_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self._critic.get_network_parameters(),
                    self.max_grad_norm,
                    error_if_nonfinite=True,
                )

                self._critic_optimizer.step()
            
            # Early stopping based on KL divergence
            if kl_mini:
                epoch_kl = np.mean(kl_mini)
                kl_divergences.append(epoch_kl)
                
                # If KL is too high, stop training on this batch
                if epoch_kl > self.kl_target * 1.5:
                    early_stop = True
        
        # Adaptive KL coefficient adjustment
        if self.adaptive_kl and kl_divergences:
            mean_kl = np.mean(kl_divergences)
            if mean_kl > self.kl_target * 2:
                # KL too high, increase penalty
                self.kl_coeff = min(self.kl_coeff * 1.5, self.kl_coeff_max)
            elif mean_kl < self.kl_target / 2:
                # KL too low, decrease penalty to allow more exploration
                self.kl_coeff = max(self.kl_coeff / 1.5, 0.01)
            kl_divergence = mean_kl
        else:
            # If we computed KL before loop, keep it; otherwise fall back to latest mean if available
            if kl_divergences:
                kl_divergence = np.mean(kl_divergences)

        # Align logging with the actual training batch (after replay buffer)
        self.smiles = sample_smiles
        score = adjusted_score

        self._timestep_report(
            prob_score,
            np.mean(critic_loss_mini),
            np.mean(actor_loss_mini),
            policy_entropy,
            kl_divergence,
            component_scores,
            component_scores_array,
            max_current_prob=max_current_prob,
            invalid_mask=invalid_mask,
        )

        if self.step % 500 == 0:
            self._logger.save_intermediate_state(self._actor, self._diversity_filter)

        self.step += 1

    def _evaluate(
        self,
        seqs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Obtains values of critic and log probabilities of actor for sequence of token ids

        Args:
            seqs (torch.Tensor): batches of sequences of token ids [n_bacthes, sequence length]

        Returns:
            tuple[torch.Tensor, torch.Tensor]: values [bacth size, sequence_length-1] and
                lob probabilities [batch size, sequence_length-1]
                for actions taken in sequence. Both excludes number for stop token.
        """

        values = self._critic.values(seqs)

        log_probs = self._actor.log_probabilities_action(seqs)

        return (values, log_probs.clamp(min=-100))

    def log_out(self):
        """Save final state of actor and memory, finalize training tracker"""
        self._logger.save_final_state(self._actor, self._diversity_filter)
        
        # Finalize training tracker - generate plots and save model
        source_log_path = os.path.join(
            self._config.logging.parameters.get("result_folder", "logs/results"), 
            "training.log"
        )
        
        # Get config as dict for saving
        try:
            config_dict = {
                "reinforcement_learning": {
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "n_steps": self.config.n_steps,
                    "specific_parameters": dict(self.config.specific_parameters),
                },
                "kl_coeff_final": self.kl_coeff,  # Save final adaptive KL coefficient
            }
        except:
            config_dict = None
        
        self.training_tracker.finalize(
            agent=self._actor,
            source_log_path=source_log_path,
            config=config_dict
        )

    def _timestep_report(
        self,
        prob_score: np.ndarray,
        critic_loss: float,
        actor_loss: float,
        policy_entropy: float,
        kl_divergence: float = 0.0,
        component_scores: dict = None,
        component_scores_array: dict = None,
        max_current_prob: float = None,
        invalid_mask: np.ndarray = None,
    ):
        """Timestep report to the standard error output, using given logger

        Args:
            prob_score (np.ndarray): raw activity probabilities (after diversity filter, before reward scaling)
            critic_loss (float): critic loss
            actor_loss (float): actor loss
            policy_entropy (float): (average) policy entropy
            kl_divergence (float): KL divergence between agent and prior
            component_scores (dict): individual component scores {name: mean_score}
            component_scores_array (dict): per-molecule component scores {name: array}
            invalid_mask (np.ndarray): mask of invalid SMILES in current batch
        """
        # Raw activity probabilities (before reward scaling)
        if invalid_mask is not None and len(invalid_mask) == len(prob_score):
            valid_mask = ~invalid_mask
        else:
            valid_mask = np.ones_like(prob_score, dtype=bool)

        if valid_mask.any():
            mean_prob = np.mean(prob_score[valid_mask])
            current_batch_max = np.max(prob_score[valid_mask])
        else:
            mean_prob = 0.0
            current_batch_max = 0.0

        # prob_max: max of newly generated molecules in current step
        prob_max = max_current_prob if max_current_prob is not None else current_batch_max
        
        # Update historical maximum across ALL training steps
        self._historical_max_prob = max(self._historical_max_prob, current_batch_max)
        
        # Calculate Top-5 average score (最高5个分子的平均分)
        valid_scores = prob_score[valid_mask] if valid_mask.any() else np.array([])
        top_k = min(5, len(valid_scores))
        if top_k > 0:
            top5_mean = np.mean(np.sort(valid_scores)[-top_k:])
        else:
            top5_mean = 0.0
        
        valid_fraction = fraction_valid_smiles(self.smiles)
        mean_len_smiles = np.mean([len(smi) for smi in self.smiles])

        # Reset mechanism: if validity drops below threshold for patience episodes, reset to prior
        if valid_fraction < self.reset_validity_threshold:
            self.n_invalid_steps += 1
        else:
            self.n_invalid_steps = 0

        if self.n_invalid_steps > self.reset_patience:
            self._logger.log_message(
                f"\n*** RESET TRIGGERED: Validity {valid_fraction:.1f}% < {self.reset_validity_threshold:.1f}% "
                f"for {self.n_invalid_steps} steps. Resetting to prior weights. ***\n"
            )
            self.reset()
            self.n_invalid_steps = 0

        # Get best molecule
        if len(prob_score) > 0 and len(self.smiles) > 0:
            best_idx = np.argmax(prob_score)
            best_smiles = self.smiles[best_idx] if best_idx < len(self.smiles) else self.smiles[0]
            best_score = prob_score[best_idx]
        else:
            best_smiles = "N/A"
            best_score = 0.0

        # Compact timestep report (2 lines)
        timestep_report = (
            f"Step {self.step:4d} | Valid: {valid_fraction:5.1f}% | "
            f"Prob S: {mean_prob:.3f} (Top5: {top5_mean:.3f}, Max: {self._historical_max_prob:.3f}) | "
            f"KL: {kl_divergence:.3f} | Ent: {policy_entropy:.2f} | "
            f"A_Loss: {actor_loss:.3f} | C_Loss: {critic_loss:.3f}\n"
            f"Best ({best_score:.3f}): {best_smiles}\n"
        )

        self._logger.log_message(timestep_report)
        
        # Log to training tracker for visualization (keeps all detailed data)
        self.training_tracker.log_step(
            step=self.step,
            total_score=mean_prob,
            component_scores=component_scores if component_scores else {},
            validity=valid_fraction,
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            policy_entropy=policy_entropy,
            kl_divergence=kl_divergence,
            avg_sequence_length=mean_len_smiles,
            kl_coeff=self.kl_coeff,
            top5_mean=top5_mean,
            max_score=self._historical_max_prob,
        )
