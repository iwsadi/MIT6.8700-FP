"""
A2C Agent for MolGPT (HuggingFace GPT-2 based molecular generator).

Uses pre-trained MolGPT for high-validity SMILES generation with RL fine-tuning.
Supports KL divergence penalty to prevent mode collapse.
"""

import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from typing import List, Tuple, Optional

from reinvent_scoring import FinalSummary

from ..utils.general import to_tensor
from ..model.critic_model import CriticModel
from ..model.actor_model_gpt2 import ActorModelGPT2
from .base_agent import BaseAgent
from ..configuration_envelope import ConfigurationEnvelope
from .utils.rewards import rewards_to_go
from .utils.sample import sample_unique_sequences


class A2CMolGPT(BaseAgent):
    """
    A2C agent using MolGPT (GPT-2) as the actor.
    
    Features:
    - Pre-trained MolGPT achieves ~99% validity out of the box
    - KL divergence penalty to stay close to prior
    - Compatible with SMILES-RL scoring functions
    """

    def __init__(
        self, 
        config: ConfigurationEnvelope, 
        scoring_function, 
        diversity_filter, 
        replay_buffer, 
        logger
    ):
        self._logger = logger
        self.config = config.reinforcement_learning.parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sp = self.config.specific_parameters
        
        # RL hyperparameters
        self.discount_factor = sp.get("discount_factor", 1.0)
        self.entropy_penalty = sp.get("entropy_penalty", 0.01)
        self.max_grad_norm = sp.get("max_grad_norm", 1.0)
        self.learning_rate_critic = sp.get("learning_rate_critic", self.config.learning_rate)
        self.learning_rate_actor = sp.get("learning_rate_actor", self.config.learning_rate)
        
        # MolGPT specific
        self.molgpt_path = sp.get("molgpt_path", "jonghyunlee/MolGPT_pretrained-by-ZINC15")
        self.max_length = sp.get("max_length", 128)
        self.temperature = sp.get("temperature", 1.0)
        
        # KL penalty
        self.kl_coef = sp.get("kl_coef", 0.1)
        self.use_kl_penalty = sp.get("use_kl_penalty", True)

        # Initialize models
        self.reset()

        self._scoring_function = scoring_function
        self._diversity_filter = diversity_filter
        self._replay_buffer = replay_buffer

        self.step = 0
        
        # Logging
        self._log_header_printed = False

    def reset(self):
        """Initialize or reset actor and critic models."""
        print(f"\n🧬 Loading MolGPT from: {self.molgpt_path}")
        
        # Load MolGPT actor
        self._actor = ActorModelGPT2.from_pretrained(
            self.molgpt_path,
            max_length=self.max_length,
            device=self.device,
            temperature=self.temperature,
        )
        self._actor.train()
        
        # Create frozen prior (copy of initial model for KL penalty)
        if self.use_kl_penalty:
            self._prior = ActorModelGPT2.from_pretrained(
                self.molgpt_path,
                max_length=self.max_length,
                device=self.device,
                temperature=self.temperature,
            )
            self._prior.eval()
            for param in self._prior.model.parameters():
                param.requires_grad = False
            print("  ✅ Prior model loaded (frozen for KL penalty)")
        else:
            self._prior = None
        
        # Actor optimizer
        self._actor_optimizer = torch.optim.Adam(
            self._actor.get_network_parameters(), 
            lr=self.learning_rate_actor
        )
        
        # Critic (using RNN-based critic from prior file)
        try:
            self._critic = CriticModel.load_from_file(
                file_path=self.config.prior, 
                sampling_mode=False
            )
            self._critic_optimizer = torch.optim.Adam(
                self._critic.get_network_parameters(), 
                lr=self.learning_rate_critic
            )
            print("  ✅ Critic loaded from prior")
        except Exception as e:
            print(f"  ⚠️ Could not load critic: {e}")
            print("  Using actor-only mode (no baseline)")
            self._critic = None
            self._critic_optimizer = None

    @torch.no_grad()
    def act(self, batch_size: int) -> List[str]:
        """Generate a batch of SMILES using the actor."""
        self._actor.eval()
        
        # Sample sequences
        self.seqs, self.smiles, self.batch_log_probs = self._actor.sample(
            batch_size=batch_size,
            temperature=self.temperature,
        )
        
        self._actor.train()
        return deepcopy(self.smiles)

    def log_out(self):
        """Save final state."""
        self._logger.save_final_state(self._actor, self._diversity_filter)

    def update(self, smiles: List[str]):
        """Update actor and critic based on rewards."""
        
        # Get scores from scoring function
        try:
            score_summary = self._scoring_function.get_final_score_for_step(smiles, self.step)
        except TypeError:
            # Fallback if scoring function doesn't support step
            score_summary = FinalSummary(
                np.zeros((len(smiles),), dtype=np.float32), 
                smiles, [], []
            )

        score_summary = deepcopy(score_summary)
        
        # Apply diversity filter
        score = self._diversity_filter.update_score(score_summary, self.step)
        raw_scores = np.array(score_summary.total_score)
        
        # Sample from replay buffer
        sample_smiles, sample_rewards = self._replay_buffer(
            score_summary.scored_smiles, score
        )
        
        # Convert to sequences
        sample_seqs = self._actor.smiles_to_sequences(sample_smiles)
        sample_rewards = to_tensor(sample_rewards).to(self.device)
        
        # Compute KL penalty if enabled
        if self.use_kl_penalty and self._prior is not None:
            kl_penalty = self._compute_kl_penalty(sample_seqs)
            augmented_rewards = sample_rewards - self.kl_coef * kl_penalty
        else:
            kl_penalty = torch.zeros_like(sample_rewards)
            augmented_rewards = sample_rewards
        
        # Compute losses and update
        critic_loss, actor_loss = self._update(sample_seqs, augmented_rewards)
        
        # Compute entropy for logging
        with torch.no_grad():
            log_probs, probs = self._actor.log_and_probabilities(sample_seqs)
            entropy = -(probs * log_probs).sum(-1).mean()
        
        # Log progress
        self._timestep_report(
            shaped_score=score,
            raw_score=raw_scores,
            critic_loss=critic_loss.item() if critic_loss is not None else 0.0,
            actor_loss=actor_loss.item(),
            entropy=entropy.item(),
            kl_penalty=kl_penalty.mean().item(),
        )
        
        # Save intermediate checkpoints
        if self.step % 250 == 0 and self.step > 0:
            self._logger.save_intermediate_state(self._actor, self._diversity_filter)

        self.step += 1

    def _compute_kl_penalty(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence penalty between actor and prior.
        
        KL = sum over tokens of (agent_log_prob - prior_log_prob)
        """
        with torch.no_grad():
            # Get log probs from actor (current policy)
            agent_log_probs = self._actor.log_probabilities_action(sequences)
            
            # Get log probs from prior (frozen initial policy)
            prior_log_probs = self._prior.log_probabilities_action(sequences)
            
            # KL penalty per token
            kl_per_token = agent_log_probs - prior_log_probs  # (batch, seq_len-1)
            
            # Mask out padding
            mask = (sequences[:, 1:] != self._actor._pad_id).float()
            
            # Sum over sequence, average over valid tokens
            kl_sum = (kl_per_token * mask).sum(dim=1)
            seq_lengths = mask.sum(dim=1).clamp(min=1)
            kl_penalty = kl_sum / seq_lengths  # (batch,)
        
        return kl_penalty

    def _update(
        self, 
        sample_seqs: torch.Tensor, 
        sample_rewards: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Perform A2C update.
        
        Returns:
            critic_loss: Critic loss (None if no critic)
            actor_loss: Actor loss
        """
        # Get action log probabilities
        self._actor.train()
        log_probs_action = self._actor.log_probabilities_action(sample_seqs)
        
        # Compute baseline (critic values) if available
        if self._critic is not None:
            values = self._critic.values(sample_seqs)
            batch_rewards = rewards_to_go(sample_seqs, sample_rewards, self.discount_factor)
            advantage = batch_rewards - values.detach()
        else:
            # No baseline - use rewards directly
            advantage = sample_rewards.unsqueeze(1).expand_as(log_probs_action)
            batch_rewards = advantage
            values = None
        
        # Actor loss: -E[log_prob * advantage]
        actor_loss = -torch.mean(log_probs_action * advantage)
        
        # Add entropy bonus for exploration
        log_probs, probs = self._actor.log_and_probabilities(sample_seqs)
        entropy = -(probs * log_probs).sum(-1).mean()
        actor_loss = actor_loss - self.entropy_penalty * entropy
        
        # Update actor
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._actor.get_network_parameters(), 
            max_norm=self.max_grad_norm
        )
        self._actor_optimizer.step()
        
        # Update critic if available
        if self._critic is not None and values is not None:
            critic_loss = 0.5 * F.mse_loss(values, batch_rewards)
            self._critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._critic.get_network_parameters(), 
                max_norm=self.max_grad_norm
            )
            self._critic_optimizer.step()
        else:
            critic_loss = None
        
        return critic_loss, actor_loss

    def _timestep_report(
        self, 
        shaped_score, 
        raw_score, 
        critic_loss, 
        actor_loss, 
        entropy,
        kl_penalty,
    ):
        """Log training progress."""
        if not self._log_header_printed:
            header = (
                f"\n{'Step':>6} | {'Shaped':>8} | {'Raw':>8} | "
                f"{'Critic':>8} | {'Actor':>8} | {'Entropy':>8} | {'KL':>8}"
            )
            self._logger.log_message(header)
            self._logger.log_message("-" * 70)
            self._log_header_printed = True
        
        msg = (
            f"{self.step:>6} | {np.mean(shaped_score):>8.4f} | {np.mean(raw_score):>8.4f} | "
            f"{critic_loss:>8.4f} | {actor_loss:>8.4f} | {entropy:>8.4f} | {kl_penalty:>8.4f}"
        )
        self._logger.log_message(msg)
        
        # Detailed log every 50 steps
        if self.step % 50 == 0:
            detail = (
                f"\n📊 Step {self.step} Summary:\n"
                f"   Shaped Score: {np.mean(shaped_score):.4f} (std: {np.std(shaped_score):.4f})\n"
                f"   Raw Score: {np.mean(raw_score):.4f}\n"
                f"   KL Penalty: {kl_penalty:.4f}\n"
                f"   Entropy: {entropy:.4f}\n"
            )
            self._logger.log_message(detail)
