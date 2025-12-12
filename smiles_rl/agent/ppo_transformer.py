"""
PPO (Proximal Policy Optimization) Agent for Transformer-based molecular generation.

Implements PPO-Clip algorithm with Generalized Advantage Estimation (GAE) for improved
training stability compared to A2C. Uses ActorModelTransformer as the policy network.

Key Features:
- PPO-Clip surrogate objective with configurable epsilon
- Generalized Advantage Estimation (GAE) with configurable lambda
- Mini-batch updates with multiple PPO epochs
- Optional entropy bonus for exploration
- Optional KL divergence penalty to frozen prior
- Early stopping based on approximate KL divergence
- Rollout buffer for efficient experience storage

References:
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Schulman et al., "High-Dimensional Continuous Control Using GAE" (2015)
"""

import numpy as np
import torch
import torch.nn.functional as F
import time
from copy import deepcopy
from typing import List, Tuple, Optional
from collections import deque
from dataclasses import dataclass, field

from ..utils.general import to_tensor
from .utils.rewards import rewards_to_go
from ..model.critic_model import CriticModel
from ..model.actor_model_transformer import ActorModelTransformer
from .base_agent import BaseAgent
from ..configuration_envelope import ConfigurationEnvelope
from reinvent_chemistry.logging import fraction_valid_smiles
from reinvent_scoring import FinalSummary
from .utils.sample import sample_unique_sequences


@dataclass
class RolloutBuffer:
    """
    Buffer for storing rollout experiences for PPO training.
    
    Stores complete trajectories including:
    - States (token sequences)
    - Actions (next tokens)
    - Log probabilities under old policy
    - Rewards
    - Values from critic
    - Masks for valid tokens
    
    Supports efficient batched retrieval and automatic cleanup.
    """
    
    states: List[torch.Tensor] = field(default_factory=list)
    actions: List[torch.Tensor] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    rewards: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    masks: List[torch.Tensor] = field(default_factory=list)
    
    def reset(self):
        """Clear all stored experiences."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.masks.clear()
    
    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        Add a batch of experiences to the buffer.
        
        Args:
            state: State tensor [batch_size, seq_len]
            action: Action tensor [batch_size, seq_len-1]
            log_prob: Log probability tensor [batch_size, seq_len-1]
            reward: Reward tensor [batch_size, seq_len-1]
            value: Value tensor [batch_size, seq_len-1]
            mask: Mask tensor [batch_size, seq_len-1] (1 for valid, 0 for padding)
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.masks.append(mask)
    
    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve all stored experiences as concatenated batches.
        
        Returns:
            Tuple of (states, actions, log_probs, rewards, values, masks)
            All tensors concatenated along batch dimension.
        """
        if len(self.states) == 0:
            raise ValueError("Buffer is empty")
        
        # Concatenate all experiences
        states = torch.cat(self.states, dim=0)
        actions = torch.cat(self.actions, dim=0)
        log_probs = torch.cat(self.log_probs, dim=0)
        rewards = torch.cat(self.rewards, dim=0)
        values = torch.cat(self.values, dim=0)
        masks = torch.cat(self.masks, dim=0)
        
        return states, actions, log_probs, rewards, values, masks
    
    def size(self) -> int:
        """Return total number of sequences in buffer."""
        return sum(s.size(0) for s in self.states)


class PPOTransformer(BaseAgent):
    """
    PPO agent using Transformer actor (ActorModelTransformer) for SMILES generation.
    
    Implements the PPO-Clip algorithm with the following key components:
    
    1. **Clipped Surrogate Objective:**
       L^CLIP(θ) = E_t[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
       where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
    
    2. **Generalized Advantage Estimation (GAE):**
       δ_t = r_t + γV(s_{t+1}) - V(s_t)
       A_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
    
    3. **Value Function Loss:**
       L^VF = (V_θ(s_t) - V_target)^2
    
    4. **Entropy Bonus:**
       L^ENT = S[π_θ](s_t)
    
    Total Loss: L = L^CLIP - c1*L^VF + c2*L^ENT
    
    Configuration Parameters:
    - clip_ratio (ε): Clipping parameter for PPO (default: 0.2)
    - ppo_epochs: Number of optimization epochs per update (default: 4)
    - mini_batch_size: Size of mini-batches for updates
    - gae_lambda (λ): Lambda for GAE (default: 0.95)
    - discount_factor (γ): Discount factor (default: 0.99)
    - value_loss_coef (c1): Value loss coefficient (default: 0.5)
    - entropy_coef (c2): Entropy coefficient (default: 0.01)
    - max_grad_norm: Maximum gradient norm for clipping (default: 0.5)
    - target_kl: Target KL divergence for early stopping (optional)
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ==================== PPO HYPERPARAMETERS ====================
        sp = self.config.specific_parameters
        
        # Core PPO parameters
        self.discount_factor = sp.get("discount_factor", 0.99)
        self.gae_lambda = sp.get("gae_lambda", 0.95)
        self.ppo_epochs = sp.get("ppo_epochs", sp.get("n_updates_per_iteration", 4))
        self.clip_ratio = sp.get("clip_ratio", sp.get("clip", 0.2))
        
        # Mini-batch configuration
        self.mini_batch_size = sp.get("mini_batch_size", None)
        self.n_minibatches = sp.get("n_minibatches", 4)
        
        # Loss coefficients
        self.value_loss_coef = sp.get("value_loss_coef", 0.5)
        self.entropy_coef = sp.get("entropy_coef", sp.get("entropy_coeff", 0.01))
        self.use_entropy_bonus = sp.get("use_entropy_bonus", True)
        
        # Gradient clipping
        self.max_grad_norm = sp.get("max_grad_norm", 0.5)
        
        # Early stopping
        self.target_kl = sp.get("target_kl", None)
        
        # ==================== KL PENALTY CONFIGURATION ====================
        # KL penalty in reward: augmented_reward = raw_score - kl_coef * KL
        self.kl_coef = sp.get("kl_coef", 0.1)
        # KL penalty in loss (separate from reward KL)
        self.kl_weight = sp.get("kl_weight", 0.0)
        
        # ==================== REWARD MODE ====================
        self.use_vanilla_rewards = sp.get("use_vanilla_rewards", True)
        
        # ==================== LEARNING RATES ====================
        self.learning_rate_actor = sp.get("learning_rate_actor", self.config.learning_rate)
        self.learning_rate_critic = sp.get("learning_rate_critic", self.config.learning_rate)
        
        # ==================== MODEL CONFIGURATION ====================
        self.transformer_weights = sp.get("transformer_weights", None)
        self.max_sequence_length_override = sp.get("max_sequence_length", None)
        self.length_schedule = sp.get("length_schedule", [[0, 160], [500, 200], [1000, 256]])
        
        # Value function clipping (PPO2 style, optional)
        self.clip_value = sp.get("clip_value", False)
        self.value_clip = sp.get("value_clip", 0.2)

        # ==================== STATE TRACKING ====================
        self._valid_history = deque(maxlen=100)
        self.rollout_buffer = RolloutBuffer()

        # Initialize models
        self.reset()

        # ==================== ENVIRONMENT COMPONENTS ====================
        self._scoring_function = scoring_function
        self._diversity_filter = diversity_filter
        self._replay_buffer = replay_buffer

        self.step = 0
        self.n_invalid_steps = 0
        self.start_time = time.time()
        
        # Log configuration
        self._log_configuration()

    def _log_configuration(self):
        """Log PPO configuration at initialization."""
        print(f"\n{'='*60}")
        print(f"PPOTransformer Configuration:")
        print(f"  clip_ratio (ε): {self.clip_ratio}")
        print(f"  ppo_epochs: {self.ppo_epochs}")
        print(f"  mini_batch_size: {self.mini_batch_size or f'auto ({self.n_minibatches} batches)'}")
        print(f"  gae_lambda (λ): {self.gae_lambda}")
        print(f"  discount_factor (γ): {self.discount_factor}")
        print(f"  value_loss_coef: {self.value_loss_coef}")
        print(f"  entropy_coef: {self.entropy_coef}")
        print(f"  max_grad_norm: {self.max_grad_norm}")
        print(f"  target_kl: {self.target_kl}")
        print(f"  use_vanilla_rewards: {self.use_vanilla_rewards}")
        print(f"  kl_coef (reward): {self.kl_coef}")
        print(f"  kl_weight (loss): {self.kl_weight}")
        print(f"{'='*60}\n")

    def reset(self) -> None:
        """Reset/initialize models and optimizers."""
        
        # ==================== CRITIC ====================
        # RNN-based critic (shared vocabulary with RNN prior)
        self._critic = CriticModel.load_from_file(
            file_path=self.config.prior, 
            sampling_mode=False
        )
        self._critic_optimizer = torch.optim.Adam(
            self._critic.get_network_parameters(),
            lr=self.learning_rate_critic,
        )

        # ==================== ACTOR ====================
        # Transformer-based policy network
        self._actor = ActorModelTransformer.load_from_file(
            pre_training_file_path=self.config.prior,
            transfer_weight_path=self.transformer_weights,
            sampling_mode=False,
        )
        self._actor_optimizer = torch.optim.Adam(
            self._actor.get_network_parameters(), 
            lr=self.learning_rate_actor
        )

        # ==================== FROZEN PRIOR ====================
        # Frozen copy of initial policy for KL regularization
        self.prior_policy = ActorModelTransformer.load_from_file(
            pre_training_file_path=self.config.prior,
            transfer_weight_path=self.transformer_weights,
            sampling_mode=True,
        )
        self.prior_policy.set_mode("inference")
        for param in self.prior_policy.get_network_parameters():
            param.requires_grad = False
        print("✓ Frozen prior policy loaded for KL regularization")

        # Override max sequence length if specified
        if self.max_sequence_length_override:
            self._actor.max_sequence_length = self.max_sequence_length_override

        # Verify vocabulary consistency
        assert (
            self._actor.get_vocabulary() == self._critic.get_vocabulary()
        ), "Actor and critic must have the same vocabulary"
        
        print(f"✓ Models initialized on {self.device}")

    def _maybe_update_max_length(self, step: int):
        """Update max sequence length based on training schedule."""
        for threshold, max_len in self.length_schedule:
            if step >= threshold:
                self._actor.max_sequence_length = max_len
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
        next_value: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        GAE provides a family of policy gradient estimators that trade off
        between bias and variance. With λ=1, it becomes standard Monte Carlo;
        with λ=0, it becomes TD(0).
        
        Formula:
            δ_t = r_t + γ * V(s_{t+1}) - V(s_t)       (TD error)
            A_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}         (GAE)
        
        Backward computation (efficient):
            A_T = δ_T
            A_t = δ_t + γλ * A_{t+1}
        
        Args:
            rewards: Per-step rewards [batch_size, seq_len-1]
            values: Value estimates [batch_size, seq_len-1]
            masks: Valid token masks [batch_size, seq_len-1] (1=valid, 0=padding)
            next_value: Bootstrap value for non-terminal states [batch_size]
            
        Returns:
            advantages: GAE advantages [batch_size, seq_len-1]
            returns: Target returns (advantages + values) [batch_size, seq_len-1]
        """
        batch_size, seq_len = rewards.shape
        
        # Bootstrap with zeros for terminal states
        if next_value is None:
            next_value = torch.zeros(batch_size, device=rewards.device)
        
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(batch_size, device=rewards.device)
        
        # Compute GAE backwards through time
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                # Last timestep: bootstrap with next_value
                next_val = next_value
            else:
                next_val = values[:, t + 1]
            
            # TD error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
            delta = rewards[:, t] + self.discount_factor * next_val - values[:, t]
            
            # GAE: A_t = δ_t + γλ*A_{t+1}
            last_gae = delta + self.discount_factor * self.gae_lambda * last_gae * masks[:, t]
            advantages[:, t] = last_gae * masks[:, t]
        
        # Returns = advantages + values (target for value function)
        returns = advantages + values
        
        return advantages, returns

    @torch.no_grad()
    def act(self, batch_size: int) -> List[str]:
        """
        Generate SMILES sequences using the current policy.
        
        Args:
            batch_size: Number of sequences to generate
            
        Returns:
            List of SMILES strings
        """
        self.seqs, self.smiles, self.batch_log_probs = sample_unique_sequences(
            self._actor, batch_size
        )
        return deepcopy(self.smiles)

    def log_out(self):
        """Save final state of actor and diversity filter."""
        self._logger.save_final_state(self._actor, self._diversity_filter)

    def _compute_kl_penalty(self, seqs: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence penalty between current policy and frozen prior.
        
        KL(π || π_prior) ≈ E[log π(a|s) - log π_prior(a|s)]
        
        Args:
            seqs: Token sequences [batch_size, seq_len]
            
        Returns:
            kl_penalty: Per-sequence average KL [batch_size]
        """
        with torch.no_grad():
            agent_log_probs = self._actor.log_probabilities_action(seqs)
            prior_log_probs = self.prior_policy.log_probabilities_action(seqs)
            
            # KL per token
            kl_per_token = agent_log_probs - prior_log_probs
            
            # Mask padding tokens (where seq == 0)
            mask = (seqs[:, 1:] != 0).float()
            
            # Average KL over valid tokens
            kl_sum = (kl_per_token * mask).sum(dim=1)
            seq_lengths = mask.sum(dim=1).clamp(min=1)
            kl_penalty = kl_sum / seq_lengths
            
        return kl_penalty

    def _compute_augmented_reward(
        self, 
        smiles: List[str], 
        raw_scores: np.ndarray,
        seqs: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute augmented reward with KL penalty.
        
        Formula: R_augmented = R_raw - kl_coef * KL(π || π_prior)
        
        Args:
            smiles: Generated SMILES strings
            raw_scores: Raw scoring function outputs [batch_size]
            seqs: Token sequences [batch_size, seq_len]
            
        Returns:
            augmented_rewards: [batch_size]
        """
        if self.use_vanilla_rewards:
            # Vanilla mode: only raw score + KL penalty
            kl_penalty = self._compute_kl_penalty(seqs)
            kl_penalty_np = kl_penalty.cpu().numpy()
            
            augmented = raw_scores - self.kl_coef * kl_penalty_np
            
            # Log KL statistics periodically
            if self.step % 50 == 0:
                print(f"  [KL] mean={kl_penalty_np.mean():.4f}, std={kl_penalty_np.std():.4f}, "
                      f"reward_adj={-self.kl_coef * kl_penalty_np.mean():.4f}")
            
            return augmented
        else:
            # Shaped mode (if needed for compatibility)
            return np.array(raw_scores, copy=True)

    def update(self, smiles: List[str]):
        """
        Perform PPO update on the policy.
        
        This method:
        1. Scores the generated SMILES
        2. Computes augmented rewards (with KL penalty)
        3. Gets samples from replay buffer
        4. Computes GAE advantages
        5. Performs multiple epochs of mini-batch PPO updates
        
        Args:
            smiles: SMILES strings generated by act()
        """
        assert (
            self._critic.get_vocabulary() == self._actor.get_vocabulary()
        ), "Actor and critic must have the same vocabulary"

        # Update max length based on schedule
        self._maybe_update_max_length(self.step)

        # ==================== SCORING ====================
        try:
            score_summary = self._scoring_function.get_final_score_for_step(
                smiles, self.step
            )
        except TypeError as e:
            print(f"Scoring error: {e}", flush=True)
            score_summary = FinalSummary(
                np.zeros((len(smiles),), dtype=np.float32), smiles, [], []
            )

        score_summary = deepcopy(score_summary)
        raw_scores = np.array(score_summary.total_score, copy=True)
        
        # Extract individual component scores if available
        self._component_scores = {}
        if hasattr(score_summary, 'scaffold_log') and score_summary.scaffold_log:
            for entry in score_summary.scaffold_log:
                try:
                    name = getattr(entry.parameters, 'name', None) or getattr(entry.parameters, 'component_type', 'unknown')
                    # ComponentSummary uses total_score or raw_score, not score
                    if hasattr(entry, 'total_score') and entry.total_score is not None:
                        self._component_scores[name] = float(np.mean(entry.total_score))
                    elif hasattr(entry, 'raw_score') and entry.raw_score is not None:
                        self._component_scores[name] = float(np.mean(entry.raw_score))
                except Exception:
                    pass  # Skip components that can't be extracted
        
        # Apply diversity filter
        _ = self._diversity_filter.update_score(score_summary, self.step)

        # Convert to sequences for KL computation
        seqs = self._actor.smiles_to_sequences(score_summary.scored_smiles)
        
        # Compute augmented rewards
        augmented_scores = self._compute_augmented_reward(
            score_summary.scored_smiles, 
            raw_scores,
            seqs
        )

        # ==================== ENTROPY MONITORING ====================
        with torch.no_grad():
            log_probs, probs = self._actor.log_and_probabilities(seqs)
            policy_entropy = -(probs * log_probs).sum(dim=2).mean()

        # ==================== REPLAY BUFFER SAMPLING ====================
        sample_smiles, sample_score = self._replay_buffer(
            score_summary.scored_smiles, augmented_scores
        )
        sample_seqs = self._actor.smiles_to_sequences(sample_smiles)
        sample_score_tensor = to_tensor(sample_score)

        # Get old log probs and values (before update)
        with torch.no_grad():
            old_values, old_log_probs = self._evaluate(sample_seqs)
        
        # ==================== CREATE MASKS AND REWARDS ====================
        # Mask: 1 for valid tokens, 0 for padding
        masks = (sample_seqs[:, 1:] != 0).float()
        
        # Per-token rewards: assign episode reward to last valid token
        batch_size, seq_len = sample_seqs.shape
        rewards = torch.zeros(batch_size, seq_len - 1, device=sample_seqs.device)
        
        for i in range(batch_size):
            valid_indices = (sample_seqs[i, :-1] != 0).nonzero(as_tuple=True)[0]
            if len(valid_indices) > 0:
                last_valid_idx = valid_indices[-1].item()
                rewards[i, last_valid_idx] = sample_score_tensor[i]
        
        # ==================== COMPUTE GAE ====================
        next_values = torch.zeros(batch_size, device=sample_seqs.device)
        advantages, returns = self.compute_gae(rewards, old_values, masks, next_values)

        n_batch = sample_seqs.size(0)
        assert n_batch > 0, "No sequences sampled"

        # Determine mini-batch size
        if self.mini_batch_size is not None:
            n_batch_train = min(self.mini_batch_size, n_batch)
        else:
            n_batch_train = max(1, n_batch // self.n_minibatches)

        # ==================== PPO UPDATE LOOP ====================
        actor_losses = []
        critic_losses = []
        kl_divs = []
        early_stop = False
        
        # Normalize advantages (more stable training)
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for epoch in range(self.ppo_epochs):
            if early_stop:
                break
                
            # Shuffle indices for mini-batch sampling
            permut = torch.randperm(n_batch)

            for start in range(0, n_batch, n_batch_train):
                end = min(start + n_batch_train, n_batch)
                mbinds = permut[start:end]

                # Get mini-batch data
                mini_seqs = sample_seqs[mbinds]
                mini_returns = returns[mbinds]
                mini_old_log_probs = old_log_probs[mbinds]
                mini_old_values = old_values[mbinds]
                mini_adv = advantages_normalized[mbinds]
                mini_masks = masks[mbinds]

                # Get current policy outputs
                values, curr_log_probs = self._evaluate(mini_seqs)

                # ==================== PPO POLICY LOSS ====================
                # Probability ratio: r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
                ratios = torch.exp(curr_log_probs - mini_old_log_probs).nan_to_num()

                # Clipped surrogate objective
                surr1 = ratios * mini_adv
                surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * mini_adv
                actor_loss = -torch.minimum(surr1, surr2)
                
                # Apply mask and compute mean
                mask_sum = mini_masks.sum().clamp(min=1)
                actor_loss = (actor_loss * mini_masks).sum() / mask_sum

                # ==================== ENTROPY BONUS ====================
                if self.use_entropy_bonus:
                    log_p, p = self._actor.log_and_probabilities(mini_seqs)
                    entropy = -(p * log_p).sum(dim=2)  # Per-position entropy
                    entropy = (entropy[:, 1:] * mini_masks).sum() / mask_sum  # Mask and average
                    actor_loss -= self.entropy_coef * entropy

                # ==================== KL REGULARIZATION IN LOSS ====================
                if self.kl_weight > 0.0:
                    with torch.no_grad():
                        prior_log_probs = self.prior_policy.log_probabilities_action(mini_seqs)
                    kl = (curr_log_probs - prior_log_probs) * mini_masks
                    kl = kl.sum() / mask_sum
                    actor_loss += self.kl_weight * kl
                    kl_divs.append(kl.item())
                    
                    # Early stopping check
                    if self.target_kl is not None and kl.item() > self.target_kl:
                        early_stop = True
                        print(f"  Early stopping at epoch {epoch}: KL={kl.item():.4f} > target={self.target_kl}")

                # ==================== VALUE LOSS ====================
                if self.clip_value:
                    # PPO2-style value clipping
                    values_clipped = mini_old_values + torch.clamp(
                        values - mini_old_values, -self.value_clip, self.value_clip
                    )
                    vf_loss1 = (values - mini_returns) ** 2
                    vf_loss2 = (values_clipped - mini_returns) ** 2
                    critic_loss = self.value_loss_coef * 0.5 * torch.maximum(vf_loss1, vf_loss2)
                else:
                    critic_loss = self.value_loss_coef * F.mse_loss(values, mini_returns, reduction='none')
                
                critic_loss = (critic_loss * mini_masks).sum() / mask_sum

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

                # ==================== GRADIENT UPDATES ====================
                # Update actor
                self._actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(
                    self._actor.get_network_parameters(),
                    self.max_grad_norm,
                    error_if_nonfinite=True,
                )
                self._actor_optimizer.step()

                # Update critic
                self._critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._critic.get_network_parameters(),
                    self.max_grad_norm,
                    error_if_nonfinite=True,
                )
                self._critic_optimizer.step()

        # ==================== LOGGING ====================
        self._timestep_report(
            raw_scores,
            augmented_scores,
            np.mean(critic_losses) if critic_losses else 0.0,
            np.mean(actor_losses) if actor_losses else 0.0,
            policy_entropy,
            np.mean(kl_divs) if kl_divs else 0.0,
        )

        # Save checkpoints
        if self.step % 250 == 0:
            self._logger.save_intermediate_state(self._actor, self._diversity_filter)

        self.step += 1

    def _evaluate(
        self,
        seqs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get critic values and actor log probabilities for sequences.
        
        Args:
            seqs: Token sequences [batch_size, seq_len]
            
        Returns:
            values: Value estimates [batch_size, seq_len-1]
            log_probs: Log probabilities of actions [batch_size, seq_len-1]
        """
        values = self._critic.values(seqs)
        log_probs = self._actor.log_probabilities_action(seqs)
        return values, log_probs.clamp(min=-20)  # Clamp for numerical stability

    def _timestep_report(
        self,
        raw_score: np.ndarray,
        augmented_score: np.ndarray,
        critic_loss: float,
        actor_loss: float,
        policy_entropy: float,
        kl_div: float = 0.0,
    ):
        """Output training metrics to logger."""
        mean_raw = np.mean(raw_score)
        mean_aug = np.mean(augmented_score)
        # fraction_valid_smiles returns percentage (0-100), NOT fraction (0-1)
        valid_pct = fraction_valid_smiles(self.smiles)
        valid_fraction = valid_pct / 100.0  # Convert to 0-1 for internal tracking
        
        # Track validity
        self._valid_history.append(valid_fraction)

        if valid_fraction < 0.8:
            self.n_invalid_steps += 1
        else:
            self.n_invalid_steps = 0

        # Auto-reset if validity drops too much
        if self.n_invalid_steps > 10:
            print(f"\n⚠️ Auto-resetting due to low validity ({self.n_invalid_steps} steps < 80%)")
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
                f"  Policy entropy: {policy_entropy:.4f} | KL div: {kl_div:.4f}\n"
            )
        else:
            timestep_report = (
                f"\n Step {self.step} | Valid: {valid_pct:4.1f}% | "
                f"Raw Score: {mean_raw:.4f} | Aug Score: {mean_aug:.4f}\n"
                f"{component_str}"
                f"  Avg SMILES length: {mean_len_smiles:.1f} ± {std_len_smiles:.1f}\n"
                f"  Critic loss: {critic_loss:.4f} | Actor loss: {actor_loss:.4f}\n"
                f"  Policy entropy: {policy_entropy:.4f} | KL div: {kl_div:.4f}\n"
            )

        self._logger.log_message(timestep_report)
        
        # Save comprehensive metrics to CSV
        self._save_metrics_csv(
            self.step, valid_pct, mean_raw, mean_aug,
            critic_loss, actor_loss, 
            policy_entropy.item() if hasattr(policy_entropy, 'item') else policy_entropy,
            mean_len_smiles, std_len_smiles, kl_div
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
        kl_div: float,
    ):
        """Save comprehensive training metrics to CSV for plotting."""
        from pathlib import Path
        import time as time_module
        
        result_folder = self._logger._log_config.result_folder if hasattr(self._logger, '_log_config') else "results_ppo_transformer"
        csv_dir = Path(result_folder)
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / "training_history.csv"
        
        # Extract DRD2 activity for main CSV
        drd2_activity = None
        if hasattr(self, '_component_scores') and self._component_scores:
            for key, value in self._component_scores.items():
                if 'drd2' in key.lower() or 'activity' in key.lower():
                    drd2_activity = value
                    break
        
        # Write header if file doesn't exist
        if not csv_path.exists():
            header = (
                "step,timestamp,validity_pct,drd2_activity,raw_score,aug_score,"
                "critic_loss,actor_loss,entropy,kl_div,"
                "mean_length,std_length\n"
            )
            with open(csv_path, 'w') as f:
                f.write(header)
        
        # Calculate elapsed time
        elapsed = time_module.time() - self.start_time
        
        # Append metrics with DRD2 activity prominently
        drd2_val = f"{drd2_activity:.6f}" if drd2_activity is not None else "NaN"
        with open(csv_path, 'a') as f:
            f.write(
                f"{step},{elapsed:.1f},{validity:.2f},{drd2_val},{raw_score:.6f},{aug_score:.6f},"
                f"{critic_loss:.6f},{actor_loss:.6f},{entropy:.6f},{kl_div:.6f},"
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

    def update_params_clip_grad_norm(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        loss: torch.Tensor,
        max_grad_norm: float = 0.5,
    ):
        """Update parameters with gradient clipping."""
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_network_parameters(), max_grad_norm)
        optimizer.step()
