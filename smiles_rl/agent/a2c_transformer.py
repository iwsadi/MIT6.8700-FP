import numpy as np
import torch
from ..utils.general import to_tensor
from .utils.rewards import rewards_to_go
from copy import deepcopy
from ..model.critic_model import CriticModel
from ..model.actor_model_transformer import ActorModelTransformer
from .base_agent import BaseAgent
from ..configuration_envelope import ConfigurationEnvelope
from reinvent_chemistry.logging import fraction_valid_smiles
from reinvent_scoring import FinalSummary
from .utils.sample import sample_unique_sequences
import time
from typing import List, Tuple
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import deque


class A2CTransformer(BaseAgent):
    """
    A2C variant that uses the Transformer actor (ActorModelTransformer) and the existing CriticModel.
    
    Reward Modes:
    - Vanilla (use_vanilla_rewards=True): raw_score - kl_coef * kl_penalty (NO shaping)
    - Shaped (use_vanilla_rewards=False): applies validity bonus, length penalty, etc.
    """

    def __init__(
        self,
        config: ConfigurationEnvelope,
        scoring_function,
        diversity_filter,
        replay_buffer,
        logger,
    ):
        self._logger = logger
        self.config = config.reinforcement_learning.parameters

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sp = self.config.specific_parameters
        self.discount_factor = sp.get("discount_factor", 1.0)
        self.use_average_network = sp.get("average_network", False)
        self.average_network_scale = sp.get("average_network_scale", 1)
        
        # Entropy penalty in loss function
        self.entropy_penalty = sp.get("entropy_penalty", 0.02)
        
        # ==================== KL PENALTY CONFIGURATION ====================
        # kl_coef: coefficient for KL penalty in REWARD augmentation
        # Formula: augmented_reward = raw_score - kl_coef * kl_penalty
        # Higher value = stronger regularization to prior (prevents mode collapse)
        # Recommended: 0.05 - 0.2 for GPT (more fragile than RNN)
        self.kl_coef = sp.get("kl_coef", 0.1)
        
        # kl_weight: coefficient for KL term in LOSS function (separate from reward)
        # Set to 0 if using kl_coef in reward to avoid double-penalizing
        self.kl_weight = sp.get("kl_weight", 0.0)
        
        # ==================== REWARD MODE ====================
        # use_vanilla_rewards: if True, use ONLY raw docking score + KL penalty
        # All shaping (validity bonus, length penalty, etc.) is DISABLED
        self.use_vanilla_rewards = sp.get("use_vanilla_rewards", True)
        
        # Legacy phase2 reward (disabled by default)
        self.use_phase2_reward = sp.get("use_phase2_reward", False)
        
        self.max_grad_norm = sp.get("max_grad_norm", 5)
        self.tau = sp.get("tau", 0.99)
        self.learning_rate_critic = sp.get("learning_rate_critic", self.config.learning_rate)
        self.learning_rate_actor = sp.get("learning_rate_actor", self.config.learning_rate)
        self.transformer_weights = sp.get("transformer_weights", "transformed_model_weights.pth")
        self.max_sequence_length_override = sp.get("max_sequence_length", None)
        # Gradual max-length schedule: list of [step_threshold, max_len]
        self.length_schedule = sp.get("length_schedule", [[0, 160], [500, 200], [1000, 256]])

        self.recent_scaffolds = deque(maxlen=2000)
        self._valid_history = deque(maxlen=100)  # rolling validity monitor

        # Initialize policy and critic parameters
        self.reset()

        self._scoring_function = scoring_function
        self._diversity_filter = diversity_filter
        self._replay_buffer = replay_buffer

        self.step = 0
        self.n_invalid_steps = 0
        self.start_time = time.time()
        
        # Log configuration
        print(f"\n{'='*60}")
        print(f"A2CTransformer Configuration:")
        print(f"  use_vanilla_rewards: {self.use_vanilla_rewards}")
        print(f"  kl_coef (reward): {self.kl_coef}")
        print(f"  kl_weight (loss): {self.kl_weight}")
        print(f"  entropy_penalty: {self.entropy_penalty}")
        print(f"{'='*60}\n")

    @torch.no_grad()
    def act(self, batch_size: int) -> List[str]:
        """Generate sequences and return corresponding SMILES strings."""
        self.seqs, self.smiles, self.batch_log_probs = sample_unique_sequences(
            self._actor, batch_size
        )
        return deepcopy(self.smiles)

    def log_out(self):
        """Save final state of actor and memory for final inspection"""
        self._logger.save_final_state(self._actor, self._diversity_filter)

    def _compute_kl_penalty(self, seqs: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sequence KL divergence penalty between agent and prior.
        
        KL penalty = agent_log_prob - prior_log_prob (per token, averaged over sequence)
        Positive when agent deviates from prior.
        
        Args:
            seqs: Token sequences [batch_size, seq_len]
            
        Returns:
            kl_penalty: Per-sequence KL penalty [batch_size]
        """
        with torch.no_grad():
            # Get agent log probabilities for actions taken
            agent_log_probs = self._actor.log_probabilities_action(seqs)  # [batch, seq_len-1]
            
            # Get prior log probabilities for same actions
            prior_log_probs = self.prior_policy.log_probabilities_action(seqs)  # [batch, seq_len-1]
            
            # KL penalty per token = agent_log_prob - prior_log_prob
            kl_per_token = agent_log_probs - prior_log_probs  # [batch, seq_len-1]
            
            # Mask out padding (where seq == 0)
            mask = (seqs[:, 1:] != 0).float()
            
            # Sum KL over sequence, normalized by sequence length
            kl_sum = (kl_per_token * mask).sum(dim=1)  # [batch]
            seq_lengths = mask.sum(dim=1).clamp(min=1)  # [batch]
            kl_penalty = kl_sum / seq_lengths  # [batch] - mean KL per token
            
        return kl_penalty

    def _compute_augmented_reward(
        self, 
        smiles: List[str], 
        raw_scores: np.ndarray,
        seqs: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute augmented reward: raw_score - kl_coef * kl_penalty
        
        Args:
            smiles: Generated SMILES strings
            raw_scores: Raw docking/activity scores [batch_size]
            seqs: Token sequences [batch_size, seq_len]
            
        Returns:
            augmented_rewards: [batch_size]
        """
        if self.use_vanilla_rewards:
            # ==================== VANILLA MODE ====================
            # ONLY raw score + KL penalty, NO shaping
            kl_penalty = self._compute_kl_penalty(seqs)
            kl_penalty_np = kl_penalty.cpu().numpy()
            
            # Augmented reward = raw_score - kl_coef * kl_penalty
            augmented = raw_scores - self.kl_coef * kl_penalty_np
            
            # Log KL stats periodically
            if self.step % 50 == 0:
                print(f"  [KL] mean={kl_penalty_np.mean():.4f}, std={kl_penalty_np.std():.4f}, "
                      f"reward_adjustment={-self.kl_coef * kl_penalty_np.mean():.4f}")
            
            return augmented
        else:
            # ==================== SHAPED MODE ====================
            # Use existing _apply_shaping (includes KL penalty)
            return self._apply_shaping(smiles, raw_scores)

    def update(self, smiles: List[str]):
        """Use Advantage Actor Critic to update policy used for sampling sequences"""

        assert (
            self._critic.get_vocabulary() == self._actor.get_vocabulary()
        ), "The actor and the critic must have the same vocabulary"

        # Gradually lift max sequence length as training progresses
        self._maybe_update_max_length(self.step)

        try:
            score_summary = self._scoring_function.get_final_score_for_step(
                smiles, self.step
            )
        except TypeError as inst:
            print(inst, flush=True)
            score_summary = FinalSummary(
                np.zeros((len(smiles),), dtype=np.float32), smiles, [], []
            )

        score_summary = deepcopy(score_summary)
        
        # Get raw scores before any modification
        raw_scores = np.array(score_summary.total_score, copy=True)
        
        # Diversity filter (optional)
        _ = self._diversity_filter.update_score(score_summary, self.step)
        
        # Convert SMILES to sequences for KL computation
        seqs_for_kl = self._actor.smiles_to_sequences(score_summary.scored_smiles)
        
        # ==================== COMPUTE AUGMENTED REWARD ====================
        # augmented_reward = raw_score - kl_coef * kl_penalty
        score = self._compute_augmented_reward(self.smiles, raw_scores, seqs_for_kl)
        score_report = deepcopy(score)

        assert len(score) == self.seqs.size(0)

        with torch.no_grad():
            seqs = self._actor.smiles_to_sequences(score_summary.scored_smiles)
            log_probabilities, probabilities = self._actor.log_and_probabilities(seqs)
            entropy = -(probabilities * log_probabilities).sum(-1)

        sample_smiles, sample_rewards = self._replay_buffer(
            score_summary.scored_smiles, score
        )
        sample_seqs = self._actor.smiles_to_sequences(sample_smiles)
        sample_rewards = to_tensor(sample_rewards)

        critic_loss, actor_loss = self._update(
            sample_seqs,
            sample_rewards,
        )

        self._timestep_report(
            score,
            score_summary.total_score, # Pass raw activity score separately
            critic_loss.item(),
            actor_loss.item(),
            entropy.mean(),
        )

        # Save checkpoints more frequently (every 250 steps)
        if self.step % 250 == 0:
            self._logger.save_intermediate_state(self._actor, self._diversity_filter)

        self.step += 1

    def _update(
        self,
        sample_seqs: torch.Tensor,
        sample_rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A2C update of actor and critic (non-shared parameters)"""

        log_probabilities, _ = self._actor.log_and_probabilities(sample_seqs)

        log_probabilities_action = log_probabilities.gather(
            -1, sample_seqs[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        values = self._critic.values(sample_seqs)

        batch_rewards = rewards_to_go(sample_seqs, sample_rewards, self.discount_factor)

        actor_loss = self.calc_actor_loss(
            sample_seqs,
            batch_rewards,
            log_probabilities_action,
            values,
        )

        # KL regularization to frozen prior (Phase 2)
        if self.kl_weight > 0.0 and hasattr(self, "prior_policy"):
            with torch.no_grad():
                prior_log_probs, _ = self.prior_policy.log_and_probabilities(sample_seqs)
            # log_probabilities already computed above
            probs = log_probabilities.exp()
            kl = (probs * (log_probabilities - prior_log_probs)).sum(dim=-1).mean()
            actor_loss += self.kl_weight * kl

        if self.use_average_network:
            log_probs_avg, probs_avg = self._actor_average.log_and_probabilities(
                sample_seqs
            )
            kl = (probs_avg * (log_probs_avg - log_probabilities)).sum() / (
                log_probabilities.size(0) + log_probabilities.size(1)
            )
            kl *= self.average_network_scale
            actor_loss += kl

        self.update_params_clip_grad_norm(
            self._actor,
            self._actor_optimizer,
            actor_loss,
            max_grad_norm=self.max_grad_norm,
        )

        if self.use_average_network:
            self._update_params_moving_average(
                self._actor_average, self._actor, self.tau
            )

        critic_loss = self.calc_critic_loss(
            sample_seqs,
            batch_rewards,
            values,
        )

        self.update_params_clip_grad_norm(
            self._critic,
            self._critic_optimizer,
            critic_loss,
            max_grad_norm=self.max_grad_norm,
        )

        return critic_loss, actor_loss

    def calc_actor_loss(
        self,
        sample_seqs: torch.Tensor,
        batch_rewards: torch.Tensor,
        log_probabilities_action: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        advantage = batch_rewards - values.detach()
        actor_loss = -torch.mean(log_probabilities_action * advantage)
        actor_loss -= self.entropy_penalty * torch.mean(
            torch.sum(torch.exp(log_probabilities_action) * log_probabilities_action, dim=-1)
        )
        return actor_loss

    def calc_critic_loss(
        self,
        sample_seqs: torch.Tensor,
        batch_rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        target = batch_rewards
        critic_loss = 0.5 * torch.nn.functional.mse_loss(target, values)
        return critic_loss

    def reset(self):
        """Reset model parameters"""
        self._actor = ActorModelTransformer.load_from_file(
            pre_training_file_path=self.config.agent,
            transfer_weight_path=self.transformer_weights,
            sampling_mode=False,
        )
        # Freeze embeddings if supported (report suggests improved stability)
        if hasattr(self._actor.transformer, "freeze_layers"):
            self._actor.transformer.freeze_layers(["embed"])
        if self.max_sequence_length_override:
            self._actor.max_sequence_length = self.max_sequence_length_override
        self._actor_optimizer = torch.optim.Adam(
            self._actor.get_network_parameters(), lr=self.learning_rate_actor
        )

        self._critic = CriticModel.load_from_file(
            file_path=self.config.prior, sampling_mode=False
        )
        self._critic_optimizer = torch.optim.Adam(
            self._critic.get_network_parameters(),
            lr=self.config.learning_rate,
        )

        # Frozen prior policy for KL regularization (Phase 2)
        self.prior_policy = ActorModelTransformer.load_from_file(
            pre_training_file_path=self.config.prior,
            transfer_weight_path=None,
            sampling_mode=True,
        )
        self._disable_gradients(self.prior_policy, use_inference_mode=True)

        if self.use_average_network:
            self._actor_average = ActorModelTransformer.load_from_file(
                pre_training_file_path=self.config.prior,
                transfer_weight_path=self.transformer_weights,
                sampling_mode=True,
            )
            self._disable_gradients(self._actor_average, use_inference_mode=True)

    def _disable_gradients(self, model, use_inference_mode: bool = False):
        if use_inference_mode and hasattr(model, "set_mode"):
            model.set_mode("inference")
        for param in model.get_network_parameters():
            param.requires_grad = False

    def _maybe_update_max_length(self, step: int):
        """Update actor max sequence length based on schedule thresholds."""
        target_len = None
        for threshold, mlen in self.length_schedule:
            if step >= threshold:
                target_len = mlen
            else:
                break
        if target_len is not None and target_len != self._actor.max_sequence_length:
            self._actor.max_sequence_length = target_len

    # Optimizer helpers
    def update_params_clip_grad_norm(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        loss: torch.Tensor,
        n_steps: int = 1,
        max_grad_norm: float = 5,
    ):
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_network_parameters(), max_grad_norm)
        for _ in range(n_steps):
            optimizer.step()

    def _update_params_moving_average(self, moving_model, model, tau):
        for moving_params, params in zip(
            moving_model.get_network_parameters(), model.get_network_parameters()
        ):
            moving_params.data.copy_(tau * params.data + (1.0 - tau) * moving_params.data)

    def save_q_table_and_probabilities(
        self,
        seqs: torch.Tensor,
    ):
        with torch.no_grad():
            values = self._critic.values(seqs)
            log_probs, probs = self._actor.log_and_probabilities(seqs)
        self._logger.save_q_table_and_probabilities(values, probs, log_probs)

    def _timestep_report(
        self,
        shaped_score: np.ndarray,
        raw_score: np.ndarray,
        critic_loss: float,
        actor_loss: float,
        entropy: torch.Tensor,
    ):
        fraction_valid_smiles_sampled = fraction_valid_smiles(self.smiles)
        
        # Calculate metrics
        mean_shaped_score = np.mean(shaped_score)
        mean_raw_score = np.mean(raw_score) 
        
        # Calculate valid-only raw score
        valid_indices = [i for i, s in enumerate(self.smiles) if Chem.MolFromSmiles(s) is not None]
        if valid_indices:
            mean_valid_raw_score = np.mean(raw_score[valid_indices])
        else:
            mean_valid_raw_score = 0.0

        lengths = [len(smi) for smi in self.smiles]
        mean_len_smiles = np.mean(lengths)
        std_len_smiles = np.std(lengths)

        if fraction_valid_smiles_sampled < 0.8:
            self.n_invalid_steps += 1
        else:
            self.n_invalid_steps = 0
        if self.n_invalid_steps > 10:
            self.reset()
            self.n_invalid_steps = 0

        # Rolling validity (0-1) for monitoring
        self._valid_history.append(fraction_valid_smiles_sampled / 100.0)
        rolling_valid = sum(self._valid_history) / len(self._valid_history)

        timestep_report = (
            f"\n Step {self.step} Valid: {fraction_valid_smiles_sampled:4.1f}% "
            f"Shaped: {mean_shaped_score:.4f} Raw(All): {mean_raw_score:.4f} Raw(Valid): {mean_valid_raw_score:.4f}\n"
            f"Length: {mean_len_smiles:.1f} +/- {std_len_smiles:.1f}\n"
            f"Critic loss: {critic_loss:.4f} Actor loss: {actor_loss:.4f} Entropy: {entropy:.4f} Rolling Valid(0-1): {rolling_valid:.3f}\n"
        )

        self._logger.log_message(timestep_report)

        # Every 100 steps, dump a compact CSV-friendly line for plotting
        if self.step % 100 == 0:
            mean_entropy = float(entropy.mean().item()) if hasattr(entropy, "mean") else float(entropy)
            report_line = (
                f"{self.step},"
                f"{fraction_valid_smiles_sampled:.4f},"
                f"{mean_shaped_score:.4f},"
                f"{mean_raw_score:.4f},"
                f"{mean_valid_raw_score:.4f}," # Add valid raw score to CSV
                f"{mean_len_smiles:.2f},"
                f"{std_len_smiles:.2f}," # Add len std to CSV
                f"{critic_loss:.6f},"
                f"{actor_loss:.6f},"
                f"{mean_entropy:.6f},"
                f"{rolling_valid:.4f}"
            )
            self._logger.log_message(f"PLOT_STEP,{report_line}")

    def _apply_shaping(self, smiles: List[str], score: np.ndarray) -> np.ndarray:
        """
        Phase-gated reward:
        - Phase 1 (default): R_temp = +1.0 if valid, -0.1 otherwise
        - Phase 2 (if self.use_phase2_reward): complex R_new with gate, diversity, and penalties
        - Vanilla (if self.use_vanilla_rewards): return raw score unchanged
        """
        if self.use_vanilla_rewards:
            return np.array(score, copy=True)

        # Phase 2: Complex R_new (keep available)
        if self.use_phase2_reward:
            shaped = np.zeros(len(smiles), dtype=np.float32)
            r_activity = np.array(score, copy=True)  # assume score contains the raw DRD2 activity

            def is_carbon_chain_fn(s):
                if len(s) == 0:
                    return False
                c_ratio = s.count("C") / len(s)
                return c_ratio > 0.85 and len(s) > 12

            # scaffold diversity bonus (uses self.recent_scaffolds)
            def scaffold_bonus(mol):
                try:
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                    if scaffold not in self.recent_scaffolds:
                        self.recent_scaffolds.append(scaffold)
                        return 0.1
                except Exception:
                    pass
                return 0.0

            for i, smi in enumerate(smiles):
                mol = Chem.MolFromSmiles(smi)
                valid = mol is not None
                carbon_chain = is_carbon_chain_fn(smi)

                # M_validity gate
                if not valid:
                    m_validity = 0.0
                elif carbon_chain:
                    m_validity = 0.1
                else:
                    m_validity = 1.0

                # R_activity
                r_act = r_activity[i] if valid else 0.0

                # R_diversity
                r_div = scaffold_bonus(mol) if valid else 0.0

                # F_penalty
                f_pen = 0.0
                length = len(smi)
                if not valid:
                    f_pen = -0.5
                elif carbon_chain:
                    f_pen = -0.2
                elif length < 10 or length > 100:
                    f_pen = -0.1

                shaped[i] = m_validity * (r_act + r_div) + f_pen

            return shaped

        # Default shaping: length window + carbon penalty + validity/hetero/ring bonuses
        shaped = np.array(score, copy=True)

        for i, smi in enumerate(smiles):
            if not smi:
                shaped[i] -= 0.2
                continue

            # Length shaping
            if len(smi) < 3:
                shaped[i] -= 0.2
            elif 5 <= len(smi) <= 80:
                shaped[i] += 0.2
            elif len(smi) > 120:
                shaped[i] -= 0.2

            # Carbon proportion penalty
            c_count = smi.count("C")
            if len(smi) > 0:
                c_ratio = c_count / len(smi)
                if c_ratio > 0.85 and len(smi) > 10:
                    shaped[i] -= 0.2

            # RDKit validity bonus + feature bonus (hetero/rings)
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                shaped[i] += 0.1
                hetero = sum(1 for a in mol.GetAtoms() if a.GetSymbol() != "C")
                if hetero > 0:
                    shaped[i] += 0.1
                rings = mol.GetRingInfo().NumRings()
                if rings > 0:
                    shaped[i] += 0.1

        return shaped
