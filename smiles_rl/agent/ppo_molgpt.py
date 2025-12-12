"""
PPO Agent for MolGPT (GPT-2 based molecular generation).

Implements PPO-Clip algorithm with GAE for MolGPT actor/critic models.
This agent is specifically designed for the HuggingFace GPT-2 based MolGPT
architecture, using matching tokenizer and vocabulary.

Key Features:
- Discounted returns with GAE
- Clipped PPO surrogate objective
- Optional entropy bonus for exploration
- Optional KL penalty to frozen prior
- Minibatch updates with multiple PPO epochs
"""

import time
from copy import deepcopy
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

from reinvent_scoring import FinalSummary
from reinvent_chemistry.logging import fraction_valid_smiles

from ..configuration_envelope import ConfigurationEnvelope
from ..model.actor_model_gpt2 import ActorModelGPT2
from ..model.critic_molgpt import CriticModelMolGPT
from ..utils.general import to_tensor
from .base_agent import BaseAgent
from .utils.rewards import rewards_to_go


class PPOMolGPT(BaseAgent):
    """
    PPO agent for MolGPT-based molecular generation.
    
    Uses the GPT-2 based MolGPT as the actor with a matching transformer critic.
    Implements full PPO-Clip with GAE for improved stability.
    
    Configuration Parameters:
    - discount_factor (γ): Discount factor for returns (default: 0.99)
    - gae_lambda (λ): Lambda for GAE (default: 0.95)
    - ppo_epochs/n_updates_per_iteration: PPO epochs per update (default: 4)
    - clip_ratio/clip: Clipping parameter ε (default: 0.2)
    - mini_batch_size: Size of mini-batches
    - n_minibatches: Number of mini-batches if mini_batch_size not specified
    - value_loss_coef: Coefficient for value loss (default: 0.5)
    - entropy_coef/entropy_coeff: Entropy bonus coefficient (default: 0.01)
    - max_grad_norm: Maximum gradient norm (default: 0.5)
    - target_kl: Target KL for early stopping (optional)
    - kl_coef: KL penalty coefficient in reward (default: 0.1)
    - use_kl_penalty: Whether to use KL penalty (default: False)
    """

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

        # ==================== PPO HYPERPARAMETERS ====================
        sp = self.config.specific_parameters
        
        # Core PPO parameters
        self.discount_factor = sp.get("discount_factor", 0.99)
        self.gae_lambda = sp.get("gae_lambda", 0.95)
        self.ppo_epochs = sp.get("ppo_epochs", sp.get("n_updates_per_iteration", 4))
        self.clip_ratio = sp.get("clip_ratio", sp.get("clip", 0.2))
        
        # Backward compatibility aliases
        self.n_updates_per_iteration = self.ppo_epochs
        self.clip = self.clip_ratio
        
        # Mini-batch configuration
        self.mini_batch_size = sp.get("mini_batch_size", None)
        self.n_minibatches = sp.get("n_minibatches", 4)

        # Loss coefficients
        self.value_loss_coef = sp.get("value_loss_coef", 0.5)
        self.entropy_coef = sp.get("entropy_coef", sp.get("entropy_coeff", 0.0))
        self.entropy_coeff = self.entropy_coef  # Backward compatibility
        self.use_entropy_bonus = sp.get("use_entropy_bonus", False)
        
        # Gradient clipping
        self.max_grad_norm = sp.get("max_grad_norm", 0.5)
        self.clip_grad_norm = sp.get("clip_grad_norm", True)
        
        # Early stopping
        self.target_kl = sp.get("target_kl", None)

        # ==================== MOLGPT SPECIFIC ====================
        self.model_path = sp.get("transformer_weights", "entropy/gpt2_zinc_87m")
        self.learning_rate = self.config.learning_rate
        self.learning_rate_critic = sp.get("learning_rate_critic", self.learning_rate)

        # ==================== KL PENALTY ====================
        self.use_kl_penalty = sp.get("use_kl_penalty", False)
        self.kl_coef = sp.get("kl_coef", 0.1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize models
        self.reset()

        # Environment components
        self._diversity_filter = diversity_filter
        self._replay_buffer = replay_buffer
        self._scoring_function = scoring_function

        self.step = 0
        self.n_invalid_steps = 0
        self.start_time = time.time()
        
        # Log configuration
        print(f"\n{'='*60}")
        print(f"PPOMolGPT Configuration:")
        print(f"  clip_ratio: {self.clip_ratio}")
        print(f"  ppo_epochs: {self.ppo_epochs}")
        print(f"  gae_lambda: {self.gae_lambda}")
        print(f"  discount_factor: {self.discount_factor}")
        print(f"  entropy_coef: {self.entropy_coef}")
        print(f"  target_kl: {self.target_kl}")
        print(f"  use_kl_penalty: {self.use_kl_penalty}")
        print(f"  kl_coef: {self.kl_coef}")
        print(f"{'='*60}\n")

    def reset(self) -> None:
        """Reset/initialize actor, critic, and optional prior models."""
        print(f"\n  Initializing PPO MolGPT models from: {self.model_path}")

        # ==================== ACTOR ====================
        self._actor = ActorModelGPT2.from_pretrained(
            self.model_path,
            device=self.device,
        )
        self._actor.train()
        self._actor_optimizer = torch.optim.Adam(
            self._actor.get_network_parameters(),
            lr=self.learning_rate,
        )

        # ==================== CRITIC ====================
        self._critic = CriticModelMolGPT.from_actor(self._actor)
        self._critic.set_mode("training")
        self._critic_optimizer = torch.optim.Adam(
            self._critic.get_network_parameters(),
            lr=self.learning_rate_critic,
        )

        # ==================== FROZEN PRIOR ====================
        if self.use_kl_penalty:
            self._prior = ActorModelGPT2.from_pretrained(
                self.model_path,
                device=self.device,
            )
            self._prior.eval()
            for p in self._prior.model.parameters():
                p.requires_grad = False
            print(f"  ✅ Prior frozen for KL penalty (kl_coef={self.kl_coef})")
        else:
            self._prior = None

        # Verify vocabulary consistency
        assert (
            self._actor.get_vocabulary() == self._critic.get_vocabulary()
        ), "Actor and critic must share vocabulary"

        print(
            f"  ✅ All models initialized | clip={self.clip_ratio} "
            f"| entropy_coef={self.entropy_coef} "
            f"| discount={self.discount_factor}"
        )

    @torch.no_grad()
    def act(self, batch_size: int) -> List[str]:
        """Generate sequences and return SMILES strings."""
        self._actor.eval()
        self.seqs, self.smiles, self.batch_log_probs = self._actor.sample(batch_size)
        self._actor.train()
        return deepcopy(self.smiles)

    def update(self, smiles: List[str]):
        """
        Perform PPO update on actor and critic.
        
        This implements the full PPO-Clip algorithm with:
        1. Score computation and optional KL penalty
        2. GAE advantage estimation
        3. Multiple epochs of mini-batch updates
        4. Clipped policy and value objectives
        """
        # ==================== SCORING ====================
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

        # Valid / invalid masks
        valid_mask = np.zeros(len(smiles), dtype=bool)
        if hasattr(score_summary, "valid_idxs"):
            for idx in score_summary.valid_idxs:
                if idx < len(valid_mask):
                    valid_mask[idx] = True
        invalid_mask = ~valid_mask

        # Store raw scores before any modification and penalize invalid upfront
        raw_total_score = np.array(score_summary.total_score, copy=True)
        raw_total_score[invalid_mask] = -5.0

        # Extract individual component scores if available
        self._component_scores = {}
        if hasattr(score_summary, 'scaffold_log') and score_summary.scaffold_log:
            for entry in score_summary.scaffold_log:
                try:
                    name = getattr(entry.parameters, 'name', None) or getattr(entry.parameters, 'component_type', 'unknown')
                    if hasattr(entry, 'total_score') and entry.total_score is not None:
                        self._component_scores[name] = float(np.mean(entry.total_score))
                    elif hasattr(entry, 'raw_score') and entry.raw_score is not None:
                        self._component_scores[name] = float(np.mean(entry.raw_score))
                except Exception:
                    pass  # Skip components that can't be extracted

        # Diversity filter acts only on valid scores
        score = self._diversity_filter.update_score(score_summary, self.step)

        # Enforce invalid penalty after diversity filtering
        score[invalid_mask] = -5.0

        # Store scores for reporting
        self._raw_total_score = raw_total_score

        # Apply KL penalty to valid rewards only
        kl_penalty_value = 0.0
        if self.use_kl_penalty and self._prior is not None:
            valid_smiles = [sm for sm, m in zip(score_summary.scored_smiles, valid_mask) if m]
            valid_scores = score[valid_mask]
            if len(valid_smiles) > 0:
                augmented_scores, kl_penalty_value = self._apply_kl_penalty_with_value(
                    valid_smiles, valid_scores
                )
                score[valid_mask] = augmented_scores

        # Policy entropy for logging
        with torch.no_grad():
            seqs = self._actor.smiles_to_sequences(score_summary.scored_smiles)
            log_probs, probs = self._actor.log_and_probabilities(seqs)
            policy_entropy = -(probs * log_probs).sum(dim=2).mean()

        # ==================== REPLAY BUFFER ====================
        sample_smiles, sample_score = self._replay_buffer(
            score_summary.scored_smiles, score
        )
        sample_seqs = self._actor.smiles_to_sequences(sample_smiles)

        # Skip if sequences too short
        if sample_seqs.size(1) <= 1:
            print("  [WARN] Skipping update: sequences have length <= 1")
            self.step += 1
            return

        # ==================== COMPUTE ADVANTAGES (GAE) ====================
        pad_id = getattr(self._actor, "_pad_id", None)
        eos_id = getattr(self._actor, "_eos_id", None)
        pad_token_id = pad_id if pad_id is not None else (eos_id if eos_id is not None else 0)
        eos_token_id = eos_id if eos_id is not None else pad_token_id

        sample_score_tensor = to_tensor(sample_score)

        # Mask for valid (non-pad/eos) action positions
        action_tokens = sample_seqs[:, 1:]
        non_terminal_mask = (action_tokens != pad_token_id) & (action_tokens != eos_token_id)

        # Reward per timestep: place sequence reward at last non-terminal token
        rewards_per_timestep = torch.zeros_like(action_tokens, dtype=torch.float32, device=self.device)
        lengths = non_terminal_mask.sum(dim=1)
        for i in range(sample_seqs.size(0)):
            if lengths[i] > 0:
                rewards_per_timestep[i, lengths[i] - 1] = sample_score_tensor[i]

        with torch.no_grad():
            values, old_log_probs = self._evaluate(sample_seqs)
            next_values = torch.cat(
                [values[:, 1:], torch.zeros(values.size(0), 1, device=values.device)],
                dim=1,
            )

        # GAE-lambda
        advantages = torch.zeros_like(values)
        gae = torch.zeros(values.size(0), device=values.device)
        gamma = self.discount_factor
        lam = self.gae_lambda
        for t in reversed(range(values.size(1))):
            not_done = non_terminal_mask[:, t].float()
            delta = rewards_per_timestep[:, t] + gamma * next_values[:, t] * not_done - values[:, t]
            gae = delta + gamma * lam * not_done * gae
            advantages[:, t] = gae

        returns = advantages + values

        n_batch = sample_seqs.size(0)
        assert n_batch > 0, "No sequences sampled"

        # Determine mini-batch size
        if self.mini_batch_size is not None:
            n_batch_train = min(self.mini_batch_size, n_batch)
        else:
            n_batch_train = max(1, n_batch // self.n_minibatches)

        # ==================== PPO UPDATE LOOP ====================
        actor_loss_history = []
        critic_loss_history = []
        early_stop = False

        for epoch in range(self.ppo_epochs):
            if early_stop:
                break
                
            permut = torch.randperm(n_batch)
            
            for start in range(0, n_batch, n_batch_train):
                end = min(start + n_batch_train, n_batch)
                mbinds = permut[start:end]

                mini_seqs = sample_seqs[mbinds]
                mini_returns = returns[mbinds]
                mini_old_log_probs = old_log_probs[mbinds]
                mini_adv = advantages[mbinds]

                # Normalize advantages
                mini_adv = (mini_adv - mini_adv.mean()) / (mini_adv.std() + 1e-8)

                # Get current policy outputs
                values, curr_log_probs = self._evaluate(mini_seqs)

                # ==================== PPO POLICY LOSS ====================
                ratios = torch.exp(curr_log_probs - mini_old_log_probs).nan_to_num()
                surr1 = ratios * mini_adv
                surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * mini_adv
                actor_loss = -torch.minimum(surr1, surr2).mean()
                actor_loss_history.append(actor_loss.item())

                # Entropy bonus
                if self.use_entropy_bonus:
                    log_probs_ent, probs_ent = self._actor.log_and_probabilities(mini_seqs)
                    entropy = (probs_ent * log_probs_ent).sum(dim=2).mean()
                    actor_loss += self.entropy_coef * entropy

                # ==================== VALUE LOSS ====================
                critic_loss = self.value_loss_coef * F.mse_loss(values, mini_returns)
                critic_loss_history.append(critic_loss.item())

                # ==================== GRADIENT UPDATES ====================
                # Actor
                self._actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self._actor.get_network_parameters(),
                        self.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                self._actor_optimizer.step()

                # Critic
                self._critic_optimizer.zero_grad()
                critic_loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self._critic.get_network_parameters(),
                        self.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                self._critic_optimizer.step()

                # Early stopping check
                if self.target_kl is not None:
                    with torch.no_grad():
                        approx_kl = ((ratios - 1) - torch.log(ratios)).mean()
                        if approx_kl > self.target_kl:
                            early_stop = True
                            print(f"  Early stopping: KL={approx_kl:.4f} > target={self.target_kl}")
                            break

        # ==================== LOGGING ====================
        self._timestep_report(
            raw_score=self._raw_total_score,
            augmented_score=score,
            critic_loss=np.mean(critic_loss_history) if critic_loss_history else 0.0,
            actor_loss=np.mean(actor_loss_history) if actor_loss_history else 0.0,
            policy_entropy=policy_entropy,
            kl_penalty=kl_penalty_value,
        )

        if self.step % 500 == 0:
            self._logger.save_intermediate_state(self._actor, self._diversity_filter)

        self.step += 1

    def _evaluate(
        self,
        seqs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get critic values and actor log-probs for sequences."""
        values = self._critic.values(seqs)
        log_probs = self._actor.log_probabilities_action(seqs)
        return values, log_probs.clamp(min=-20)

    def _apply_kl_penalty(
        self,
        smiles: List[str],
        scores: np.ndarray,
    ) -> np.ndarray:
        """Apply KL penalty to rewards to prevent mode collapse."""
        augmented, _ = self._apply_kl_penalty_with_value(smiles, scores)
        return augmented
    
    def _apply_kl_penalty_with_value(
        self,
        smiles: List[str],
        scores: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Apply KL penalty to rewards and return the mean KL value."""
        with torch.no_grad():
            seqs = self._actor.smiles_to_sequences(smiles)
            
            if seqs.size(1) <= 1:
                return scores, 0.0
                
            agent_log_probs = self._actor.log_probabilities_action(seqs)
            prior_log_probs = self._prior.log_probabilities_action(seqs)

            kl_per_token = agent_log_probs - prior_log_probs
            pad_id = getattr(self._actor, '_pad_id', 0) or getattr(self._actor, '_eos_id', 0)
            mask = (seqs[:, 1:] != pad_id).float()

            kl_sum = (kl_per_token * mask).sum(dim=1)
            seq_lengths = mask.sum(dim=1).clamp(min=1)
            kl_penalty = (kl_sum / seq_lengths).cpu().numpy()

        augmented_scores = scores - self.kl_coef * kl_penalty
        mean_kl = float(kl_penalty.mean())
        print(
            f"  [KL] mean={mean_kl:.4f}, std={kl_penalty.std():.4f}, "
            f"reward_adj={-self.kl_coef * mean_kl:.4f}"
        )
        return augmented_scores, mean_kl

    def log_out(self):
        """Save final state."""
        self._logger.save_final_state(self._actor, self._diversity_filter)

    def _timestep_report(
        self,
        raw_score: np.ndarray,
        augmented_score: np.ndarray,
        critic_loss: float,
        actor_loss: float,
        policy_entropy: float,
        kl_penalty: float = 0.0,
    ):
        """Log step metrics with detailed breakdown."""
        mean_raw = np.mean(raw_score)
        mean_aug = np.mean(augmented_score)
        
        # fraction_valid_smiles returns percentage (0-100), NOT fraction (0-1)
        valid_pct = fraction_valid_smiles(self.smiles)
        valid_fraction = valid_pct / 100.0  # Convert to 0-1 for internal checks

        if valid_fraction < 0.8:
            self.n_invalid_steps += 1
        else:
            self.n_invalid_steps = 0

        if self.n_invalid_steps > 10:
            print(f"  ⚠️ Resetting due to {self.n_invalid_steps} invalid steps")
            self.reset()
            self.n_invalid_steps = 0

        mean_len_smiles = np.mean([len(smi) for smi in self.smiles])
        std_len_smiles = np.std([len(smi) for smi in self.smiles])
        
        # Extract DRD2 activity score prominently
        drd2_activity = None
        if hasattr(self, '_component_scores') and self._component_scores:
            # Look for DRD2 activity (case-insensitive)
            for key, value in self._component_scores.items():
                if 'drd2' in key.lower() or 'activity' in key.lower():
                    drd2_activity = value
                    break
        
        # Build component scores string
        component_str = ""
        if hasattr(self, '_component_scores') and self._component_scores:
            parts = [f"{k}: {v:.4f}" for k, v in self._component_scores.items()]
            component_str = f"  Components: {', '.join(parts)}\n"
        
        # Prominently display DRD2 activity
        if drd2_activity is not None:
            timestep_report = (
                f"\n Step {self.step} | Valid: {valid_pct:4.1f}% | "
                f"🎯 DRD2 Activity: {drd2_activity:.4f} | "
                f"Raw Score: {mean_raw:.4f} | Aug Score: {mean_aug:.4f}\n"
                f"{component_str}"
                f"  Avg SMILES length: {mean_len_smiles:.1f} ± {std_len_smiles:.1f}\n"
                f"  Critic loss: {critic_loss:.4f} | Actor loss: {actor_loss:.4f}\n"
                f"  Policy entropy: {policy_entropy:.4f} | KL penalty: {kl_penalty:.4f}\n"
            )
        else:
            timestep_report = (
                f"\n Step {self.step} | Valid: {valid_pct:4.1f}% | Raw Score: {mean_raw:.4f} | Aug Score: {mean_aug:.4f}\n"
                f"{component_str}"
                f"  Avg SMILES length: {mean_len_smiles:.1f} ± {std_len_smiles:.1f}\n"
                f"  Critic loss: {critic_loss:.4f} | Actor loss: {actor_loss:.4f}\n"
                f"  Policy entropy: {policy_entropy:.4f} | KL penalty: {kl_penalty:.4f}\n"
            )
        self._logger.log_message(timestep_report)

        # Save comprehensive metrics to CSV
        self._save_metrics_csv(
            step=self.step,
            validity=valid_pct,
            raw_score=mean_raw,
            aug_score=mean_aug,
            critic_loss=critic_loss,
            actor_loss=actor_loss,
            entropy=policy_entropy.item() if hasattr(policy_entropy, "item") else policy_entropy,
            mean_length=mean_len_smiles,
            std_length=std_len_smiles,
            kl_penalty=kl_penalty,
        )

    def _save_metrics_csv(
        self,
        step: int,
        validity: float,
        raw_score: float,
        aug_score: float,
        critic_loss: float,
        actor_loss: float,
        entropy: float,
        mean_length: float,
        std_length: float,
        kl_penalty: float,
    ):
        """Save comprehensive metrics to CSV for plotting."""
        from pathlib import Path
        import time as time_module

        csv_dir = Path("results_ppo_molgpt")
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / "training_history.csv"

        # Extract DRD2 activity for main CSV
        drd2_activity = None
        if hasattr(self, '_component_scores') and self._component_scores:
            for key, value in self._component_scores.items():
                if 'drd2' in key.lower() or 'activity' in key.lower():
                    drd2_activity = value
                    break

        # Create header if file doesn't exist
        if not csv_path.exists():
            header = (
                "step,timestamp,validity_pct,drd2_activity,raw_score,aug_score,"
                "critic_loss,actor_loss,entropy,kl_penalty,"
                "mean_length,std_length\n"
            )
            with open(csv_path, "w") as f:
                f.write(header)

        # Calculate elapsed time
        elapsed = time_module.time() - self.start_time
        
        # Write main metrics with DRD2 activity prominently
        drd2_val = f"{drd2_activity:.6f}" if drd2_activity is not None else "NaN"
        with open(csv_path, "a") as f:
            f.write(
                f"{step},{elapsed:.1f},{validity:.2f},{drd2_val},{raw_score:.6f},{aug_score:.6f},"
                f"{critic_loss:.6f},{actor_loss:.6f},{entropy:.6f},{kl_penalty:.6f},"
                f"{mean_length:.2f},{std_length:.2f}\n"
            )
        
        # Also save component scores if available
        if hasattr(self, '_component_scores') and self._component_scores:
            comp_path = csv_dir / "component_scores.csv"
            if not comp_path.exists():
                header = "step," + ",".join(self._component_scores.keys()) + "\n"
                with open(comp_path, "w") as f:
                    f.write(header)
            
            values = ",".join([f"{v:.6f}" for v in self._component_scores.values()])
            with open(comp_path, "a") as f:
                f.write(f"{step},{values}\n")
