"""
REINFORCE agent for MolGPT (GPT-2 based molecular generation).

Uses policy gradient without a critic, which avoids vocabulary mismatch issues.
This is the standard approach for GPT-based RL in molecular generation.
"""

import torch
import numpy as np
from copy import deepcopy
from typing import List, Tuple
from reinvent_scoring import FinalSummary

from ..utils.general import to_tensor
from .base_agent import BaseAgent
from ..configuration_envelope import ConfigurationEnvelope

try:
    from ..model.actor_model_gpt2 import ActorModelGPT2
except ImportError as e:
    raise ImportError("ActorModelGPT2 is required. Make sure actor_model_gpt2.py is present.") from e


class ReinforceMolGPT(BaseAgent):
    """
    REINFORCE agent for MolGPT.
    
    Uses policy gradient with baseline subtraction (average reward).
    No critic needed, avoiding vocabulary mismatch issues.
    """

    def __init__(self, config: ConfigurationEnvelope, scoring_function, diversity_filter, replay_buffer, logger):
        self._logger = logger
        self.config = config.reinforcement_learning.parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sp = self.config.specific_parameters
        self.entropy_coef = sp.get("entropy_coef", 0.01)
        self.max_grad_norm = sp.get("max_grad_norm", 1.0)
        self.learning_rate = sp.get("learning_rate_actor", self.config.learning_rate)
        self.transformer_weights = sp.get("transformer_weights", "entropy/gpt2_zinc_87m")
        self.baseline_momentum = sp.get("baseline_momentum", 0.99)
        
        # Running baseline for variance reduction
        self.baseline = 0.0
        
        self.reset()

        self._scoring_function = scoring_function
        self._diversity_filter = diversity_filter
        self._replay_buffer = replay_buffer

        self.step = 0

    @torch.no_grad()
    def act(self, batch_size: int) -> List[str]:
        """Generate SMILES using MolGPT."""
        self.seqs, self.smiles, self.batch_log_probs = self._actor.sample(batch_size)
        return deepcopy(self.smiles)

    def log_out(self):
        self._logger.save_final_state(self._actor, self._diversity_filter)

    def update(self, smiles: List[str]):
        """Update policy using REINFORCE with baseline."""
        
        # Get scores
        try:
            score_summary = self._scoring_function.get_final_score_for_step(smiles, self.step)
        except TypeError:
            score_summary = FinalSummary(np.zeros((len(smiles),), dtype=np.float32), smiles, [], [])

        score_summary = deepcopy(score_summary)
        score = self._diversity_filter.update_score(score_summary, self.step)
        
        # Get sequences and rewards from replay buffer
        sample_smiles, sample_rewards = self._replay_buffer(score_summary.scored_smiles, score)
        sample_rewards = np.array(sample_rewards, dtype=np.float32)
        
        # Update baseline (moving average)
        batch_mean_reward = np.mean(sample_rewards)
        self.baseline = self.baseline_momentum * self.baseline + (1 - self.baseline_momentum) * batch_mean_reward
        
        # Convert to tensors
        sample_seqs = self._actor.smiles_to_sequences(sample_smiles)
        rewards_tensor = torch.tensor(sample_rewards - self.baseline, device=self.device)
        
        # Compute log probabilities
        self._actor.train()
        log_probs = self._actor.log_probabilities_action(sample_seqs)  # (batch, seq_len-1)
        
        # Create mask for non-padding tokens
        # Assuming PAD token has ID 1 for MolGPT
        pad_id = self._actor._pad_id if hasattr(self._actor, '_pad_id') else 1
        mask = (sample_seqs[:, 1:] != pad_id).float()
        
        # Compute sequence-level log probability (sum over tokens)
        seq_log_probs = (log_probs * mask).sum(dim=1)  # (batch,)
        
        # REINFORCE loss: -reward * log_prob
        policy_loss = -(rewards_tensor * seq_log_probs).mean()
        
        # Entropy bonus for exploration
        if self.entropy_coef > 0:
            # Approximate entropy from log_probs
            entropy = -(log_probs.exp() * log_probs * mask).sum(dim=1).mean()
            total_loss = policy_loss - self.entropy_coef * entropy
        else:
            total_loss = policy_loss
            entropy = torch.tensor(0.0)
        
        # Backward pass
        self._optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._actor.model.parameters(), self.max_grad_norm)
        
        self._optimizer.step()
        self._actor.eval()
        
        # Logging
        self._timestep_report(
            score, 
            score_summary.total_score, 
            policy_loss.item(), 
            entropy.item(),
            batch_mean_reward,
        )

        if self.step % 250 == 0:
            self._logger.save_intermediate_state(self._actor, self._diversity_filter)

        self.step += 1

    def reset(self):
        """Initialize/reset the actor."""
        self._actor = ActorModelGPT2.from_pretrained(self.transformer_weights, device=self.device)
        self._optimizer = torch.optim.Adam(self._actor.get_network_parameters(), lr=self.learning_rate)
        self._actor.eval()
        self.baseline = 0.0

    def _timestep_report(self, shaped_score, raw_score, policy_loss, entropy, baseline):
        """Log training progress."""
        validity = sum(1 for s in self.smiles if self._is_valid_smiles(s)) / len(self.smiles) * 100
        
        msg = (
            f"\n Step {self.step} | "
            f"Shaped: {np.mean(shaped_score):.4f} | "
            f"Raw: {np.mean(raw_score):.4f} | "
            f"Baseline: {baseline:.4f}\n"
            f"Policy Loss: {policy_loss:.4f} | "
            f"Entropy: {entropy:.4f} | "
            f"Validity: {validity:.1f}%\n"
        )
        self._logger.log_message(msg)
    
    def _is_valid_smiles(self, smiles: str) -> bool:
        """Check if SMILES is valid using RDKit."""
        try:
            from rdkit import Chem
            return Chem.MolFromSmiles(smiles) is not None
        except:
            return False
