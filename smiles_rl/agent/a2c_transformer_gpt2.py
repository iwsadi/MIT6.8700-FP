"""
Experimental A2C agent using a GPT-2 prior (entropy/gpt2_zinc_87m).
This is intentionally isolated to avoid breaking existing Transformer/RNN flows.

Note:
- Requires transformers and internet/cache access to load the model/tokenizer.
- Sampling/generation is not fully wired; this is a scaffold to be extended.
"""

import torch
import numpy as np
from copy import deepcopy
from typing import List, Tuple
from reinvent_scoring import FinalSummary

from ..utils.general import to_tensor
from ..model.critic_model import CriticModel
from .base_agent import BaseAgent
from ..configuration_envelope import ConfigurationEnvelope
from .utils.rewards import rewards_to_go
from .utils.sample import sample_unique_sequences

try:
    from ..model.actor_model_gpt2 import ActorModelGPT2
except ImportError as e:
    raise ImportError("ActorModelGPT2 is required. Make sure actor_model_gpt2.py is present.") from e


class A2CTransformerGPT2(BaseAgent):
    """
    A2C variant that uses a GPT-2 LM head as actor (entropy/gpt2_zinc_87m).
    Sampling is stubbed; extend ActorModelGPT2.sample for full functionality.
    """

    def __init__(self, config: ConfigurationEnvelope, scoring_function, diversity_filter, replay_buffer, logger):
        self._logger = logger
        self.config = config.reinforcement_learning.parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sp = self.config.specific_parameters
        self.discount_factor = sp.get("discount_factor", 1.0)
        self.entropy_penalty = sp.get("entropy_penalty", 0.01)
        self.max_grad_norm = sp.get("max_grad_norm", 5)
        self.learning_rate_critic = sp.get("learning_rate_critic", self.config.learning_rate)
        self.learning_rate_actor = sp.get("learning_rate_actor", self.config.learning_rate)
        self.transformer_weights = sp.get("transformer_weights", "entropy/gpt2_zinc_87m")

        self.reset()

        self._scoring_function = scoring_function
        self._diversity_filter = diversity_filter
        self._replay_buffer = replay_buffer

        self.step = 0

    @torch.no_grad()
    def act(self, batch_size: int) -> List[str]:
        self.seqs, self.smiles, self.batch_log_probs = sample_unique_sequences(
            self._actor, batch_size
        )
        return deepcopy(self.smiles)

    def log_out(self):
        self._logger.save_final_state(self._actor, self._diversity_filter)

    def update(self, smiles: List[str]):
        # Skip vocab equality check for GPT-2 actor (vocab structures differ)

        try:
            score_summary = self._scoring_function.get_final_score_for_step(smiles, self.step)
        except TypeError:
            score_summary = FinalSummary(np.zeros((len(smiles),), dtype=np.float32), smiles, [], [])

        score_summary = deepcopy(score_summary)
        score = self._diversity_filter.update_score(score_summary, self.step)
        score_report = deepcopy(score)

        assert len(score) == self.seqs.size(0)

        with torch.no_grad():
            seqs = self._actor.smiles_to_sequences(score_summary.scored_smiles)
            log_probabilities, probabilities = self._actor.log_and_probabilities(seqs)
            entropy = -(probabilities * log_probabilities).sum(-1)

        sample_smiles, sample_rewards = self._replay_buffer(score_summary.scored_smiles, score)
        sample_seqs = self._actor.smiles_to_sequences(sample_smiles)
        sample_rewards = to_tensor(sample_rewards)

        critic_loss, actor_loss = self._update(sample_seqs, sample_rewards)

        self._timestep_report(score, score_summary.total_score, critic_loss.item(), actor_loss.item(), entropy.mean())

        if self.step % 250 == 0:
            self._logger.save_intermediate_state(self._actor, self._diversity_filter)

        self.step += 1

    def _update(self, sample_seqs: torch.Tensor, sample_rewards: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_probabilities, _ = self._actor.log_and_probabilities(sample_seqs)
        log_probabilities_action = log_probabilities.gather(-1, sample_seqs[:, 1:].unsqueeze(-1)).squeeze(-1)
        values = self._critic.values(sample_seqs)
        batch_rewards = rewards_to_go(sample_seqs, sample_rewards, self.discount_factor)

        advantage = batch_rewards - values.detach()
        actor_loss = -torch.mean(log_probabilities_action * advantage)
        actor_loss -= self.entropy_penalty * torch.mean(
            torch.sum(torch.exp(log_probabilities_action) * log_probabilities_action, dim=-1)
        )

        self.update_params_clip_grad_norm(self._actor, self._actor_optimizer, actor_loss, max_grad_norm=self.max_grad_norm)

        critic_loss = self.calc_critic_loss(sample_seqs, batch_rewards, values)
        self.update_params_clip_grad_norm(self._critic, self._critic_optimizer, critic_loss, max_grad_norm=self.max_grad_norm)

        return critic_loss, actor_loss

    def calc_critic_loss(self, sample_seqs: torch.Tensor, batch_rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        target = batch_rewards
        return 0.5 * torch.nn.functional.mse_loss(target, values)

    def reset(self):
        self._actor = ActorModelGPT2.from_pretrained(self.transformer_weights)
        self._actor_optimizer = torch.optim.Adam(self._actor.get_network_parameters(), lr=self.learning_rate_actor)

        self._critic = CriticModel.load_from_file(file_path=self.config.prior, sampling_mode=False)
        self._critic_optimizer = torch.optim.Adam(self._critic.get_network_parameters(), lr=self.config.learning_rate)

    def _timestep_report(self, shaped_score, raw_score, critic_loss, actor_loss, entropy):
        msg = (
            f"\n Step {self.step} "
            f"Shaped: {np.mean(shaped_score):.4f} Raw: {np.mean(raw_score):.4f}\n"
            f"Critic loss: {critic_loss:.4f} Actor loss: {actor_loss:.4f} Entropy: {entropy:.4f}\n"
        )
        self._logger.log_message(msg)

