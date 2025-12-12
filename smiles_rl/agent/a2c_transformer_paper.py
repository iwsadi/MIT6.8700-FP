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
import os

class A2CTransformerPaper(BaseAgent):
    """
    A2C variant that uses the Transformer actor (ActorModelTransformer) mimicking the paper's approach.
    - No complex reward shaping (rely on Diversity Filter).
    - Simple invalid penalty (0 or -1).
    - Batch size 128 (configured in JSON).
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
        self.entropy_penalty = sp.get("entropy_penalty", 0.01) # Typically lower for stability if not shaping heavily
        self.max_grad_norm = sp.get("max_grad_norm", 5)
        self.tau = sp.get("tau", 0.99)
        self.learning_rate_critic = sp.get("learning_rate_critic", self.config.learning_rate)
        self.learning_rate_actor = sp.get("learning_rate_actor", self.config.learning_rate)
        self.transformer_weights = sp.get("transformer_weights", "transformed_model_weights.pth")
        self.max_sequence_length_override = sp.get("max_sequence_length", None)
        
        # Initialize policy and critic parameters
        self.reset()

        self._scoring_function = scoring_function
        self._diversity_filter = diversity_filter
        self._replay_buffer = replay_buffer

        self.step = 0
        self.n_invalid_steps = 0
        self.start_time = time.time()

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

    def update(self, smiles: List[str]):
        """Use Advantage Actor Critic to update policy used for sampling sequences"""

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

        score_summary = deepcopy(score_summary)
        
        # Paper Approach: Rely on Diversity Filter to handle rewards
        score = self._diversity_filter.update_score(score_summary, self.step)
        
        # Simple Invalid Penalty (Paper says 0 for on-policy, but Scoring Function likely returns 0)
        # We ensure it's 0.
        # If we want -1 for off-policy, but we are on-policy A2C. 
        # Some implementations give small negative for invalid to encourage validity.
        # Let's stick to 0 as per "Paper 1 setup" usually implies standard REINVENT behavior.
        # However, checking valid fraction is good for logging.
        
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
            critic_loss.item(),
            actor_loss.item(),
            entropy.mean(),
        )

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
        score: np.ndarray,
        critic_loss: float,
        actor_loss: float,
        entropy: torch.Tensor,
    ):
        fraction_valid_smiles_sampled = fraction_valid_smiles(self.smiles)
        mean_score = np.mean(score)
        
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

        timestep_report = (
            f"\n Step {self.step} Valid: {fraction_valid_smiles_sampled:4.1f}% Score: {mean_score:.4f} "
            f"Length: {mean_len_smiles:.1f} +/- {std_len_smiles:.1f}\n"
            f"Critic loss: {critic_loss:.4f} Actor loss: {actor_loss:.4f} Entropy: {entropy:.4f}\n"
        )

        self._logger.log_message(timestep_report)
        
        # Explicitly save to a dedicated CSV file for easy plotting
        csv_path = "transformer_paper_v2_history.csv"
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            header = "step,valid,score,length,std_length,critic,actor,entropy\n"
            if not file_exists:
                f.write(header)
            
            # Every 10 steps, dump CSV line
            if self.step % 10 == 0:
                mean_entropy = float(entropy.mean().item()) if hasattr(entropy, "mean") else float(entropy)
                line = f"{self.step},{fraction_valid_smiles_sampled:.4f},{mean_score:.4f},{mean_len_smiles:.2f},{std_len_smiles:.2f},{critic_loss:.6f},{actor_loss:.6f},{mean_entropy:.6f}\n"
                f.write(line)

