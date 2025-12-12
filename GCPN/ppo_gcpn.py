import os
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from reinvent_scoring import FinalSummary
from reinvent_chemistry.logging import fraction_valid_smiles

from .base_agent import BaseAgent
from ..configuration_envelope import ConfigurationEnvelope

from ..utils.general import _set_torch_device

# GCPN modules (repo-local, NOT torchdrug)
from GCPN.gcpn_model import GCPNPolicy
from GCPN.create_trajectory import MoleculeTrajectory, ATOM_TYPES, BOND_TYPES
from GCPN.generate_gcpn import generate_molecule  # generation routine uses the actor policy


def _bond_type_to_index(bond_type) -> int:
    if bond_type in BOND_TYPES:
        return BOND_TYPES.index(bond_type)
    return len(BOND_TYPES)  # "other"


def _atom_type_to_index(atomic_num: int) -> int:
    if atomic_num in ATOM_TYPES:
        return ATOM_TYPES.index(atomic_num)
    return len(ATOM_TYPES)  # "other"


def _log_prob_bernoulli_from_logit(logit: torch.Tensor, target01: int) -> torch.Tensor:
    """
    Stable log-prob for Bernoulli with logits.
    """
    # log p(y=1) = -softplus(-logit), log p(y=0) = -softplus(logit)
    if target01 == 1:
        return -F.softplus(-logit)
    return -F.softplus(logit)


def _rtg_for_trajectory(score: float, n_steps: int, gamma: float) -> torch.Tensor:
    """
    Reward-to-go for an episodic reward paid at the end, matching smiles_rl/agent/utils/rewards.py:
    - for steps 0..T-2: gamma^(T-2-t)*score
    - for the terminal stop action (t=T-1): 0
    """
    if n_steps <= 0:
        return torch.zeros((0,), dtype=torch.float32)
    if n_steps == 1:
        return torch.zeros((1,), dtype=torch.float32)
    exponents = torch.arange(n_steps - 1 - 1, -1, -1, dtype=torch.float32)  # (T-2 .. 0)
    rtg = (gamma ** exponents) * float(score)
    rtg = torch.cat([rtg, torch.zeros((1,), dtype=torch.float32)], dim=0)
    return rtg


@dataclass
class StepRecord:
    data: "torch_geometric.data.Data"
    action: dict
    rtg: float
    old_logprob: float


class _GCPNActorWrapper:
    """
    Small wrapper to be compatible with the existing logger interface (save_to_file).
    """

    def __init__(self, model: GCPNPolicy):
        self.model = model

    def get_network_parameters(self):
        return self.model.parameters()

    def save_to_file(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    @staticmethod
    def load_from_file(file_path: str, input_dim: int, hidden_dim: int, num_atom_types: int, device):
        model = GCPNPolicy(input_dim, hidden_dim, num_atom_types).to(device)
        model.load_state_dict(torch.load(file_path, map_location=device))
        return _GCPNActorWrapper(model)


class _GCPNCritic(nn.Module):
    """
    Critic network for graph states.
    Uses a GCPNPolicy backbone for node embeddings, then a value head on pooled graph embedding.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_atom_types: int):
        super().__init__()
        self.backbone = GCPNPolicy(input_dim, hidden_dim, num_atom_types)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch) -> torch.Tensor:
        out = self.backbone(batch, new_node_indices=None, focus_node_indices=None)
        node_embeddings = out["node_embeddings"]
        batch_size = batch.num_graphs
        if node_embeddings.numel() == 0:
            graph_emb = torch.zeros((batch_size, self.backbone.hidden_dim), device=batch.x.device)
        else:
            graph_emb = global_mean_pool(node_embeddings, batch.batch, size=batch_size)
        v = self.value_head(graph_emb).squeeze(-1)
        return v


class PPOGCPN(BaseAgent):
    """
    PPO agent that uses the repo's GCPN graph policy (GCPNPolicy) instead of the RNN ActorModel.

    Important design choice:
    - This mirrors PPO.py's *teacher-forcing* update style: we evaluate log-probabilities on
      trajectories derived from SMILES via MoleculeTrajectory (not torchdrug).
    - act() still samples SMILES using the policy via GCPN.generate_gcpn.generate_molecule.
    """

    def __init__(
        self,
        config: ConfigurationEnvelope,
        scoring_function,
        diversity_filter,
        replay_buffer,
        logger,
    ) -> None:
        super().__init__(config, scoring_function, diversity_filter, replay_buffer, logger)

        self._config = config
        self._logger = logger
        self.config = config.reinforcement_learning.parameters
        params = self.config.specific_parameters

        self.discount_factor = params.get("discount_factor", 0.99)
        self.n_updates_per_iteration = params.get("n_updates_per_iteration", 5)
        self.clip = params.get("clip", 0.2)
        self.use_entropy_bonus = params.get("use_entropy_bonus", False)
        self.entropy_coeff = params.get("entropy_coeff", 0.001)
        self.max_grad_norm = params.get("max_grad_norm", 0.5)
        self.n_minibatches = params.get("n_minibatches", 4)

        # GCPN architecture params (must match pretraining)
        self.hidden_dim = params.get("gcpn_hidden_dim", 64)
        self.input_dim = len(ATOM_TYPES) + 1
        self.num_atom_types = len(ATOM_TYPES) + 1

        # Sampling params
        self.max_gen_steps = params.get("gcpn_max_gen_steps", 80)
        self.gen_attempts = params.get("gcpn_gen_attempts", 50)

        # Device (run.py sets default, but we keep explicit device for safety)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _set_torch_device(str(self.device))

        self.step = 0
        self.n_invalid_steps = 0
        self.start_time = time.time()

        self._load_models()

    def _load_models(self):
        # Actor weights
        if not os.path.exists(self.config.agent):
            raise FileNotFoundError(f"GCPN actor weights not found: {self.config.agent}")
        if not os.path.exists(self.config.prior):
            raise FileNotFoundError(f"GCPN critic prior weights not found: {self.config.prior}")

        self._actor = _GCPNActorWrapper.load_from_file(
            self.config.agent, self.input_dim, self.hidden_dim, self.num_atom_types, self.device
        )
        # Critic (load backbone from prior)
        self._critic = _GCPNCritic(self.input_dim, self.hidden_dim, self.num_atom_types).to(self.device)
        prior_sd = torch.load(self.config.prior, map_location=self.device)
        # prior may be a plain state_dict from GCPNPolicy
        self._critic.backbone.load_state_dict(prior_sd, strict=False)

        self._actor_optimizer = torch.optim.Adam(self._actor.get_network_parameters(), lr=self.config.learning_rate)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=self.config.learning_rate)

    @torch.no_grad()
    def act(self, batch_size: int) -> List[str]:
        self._actor.model.eval()
        smiles: List[str] = []
        seen = set()

        # Keep sampling until we get enough unique molecules or exhaust attempts.
        attempts = 0
        while len(smiles) < batch_size and attempts < self.gen_attempts * batch_size:
            attempts += 1
            mol = generate_molecule(self._actor.model, max_steps=self.max_gen_steps, device=self.device)
            if mol is None:
                continue
            smi = Chem.MolToSmiles(mol)
            if not smi or smi in seen:
                continue
            seen.add(smi)
            smiles.append(smi)

        self.smiles = deepcopy(smiles)
        return deepcopy(smiles)

    def _evaluate_step(self, data, action: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (value, log_prob, entropy_estimate) for a single step.
        """
        # Build a batch of one graph
        batch = Batch.from_data_list([data]).to(self.device)

        focus = getattr(data, "focus_node", None)
        focus_val = int(focus) if focus is not None else -1

        # Avoid passing focus indices for empty graphs
        if data.num_nodes == 0 or focus_val < 0:
            focus_node_indices = None
            new_node_indices = None
        else:
            focus_node_indices = torch.tensor([focus_val], dtype=torch.long, device=self.device)
            # In this implementation/training, new_node_indices is also set to focus
            new_node_indices = focus_node_indices

        out = self._actor.model(batch, new_node_indices=new_node_indices, focus_node_indices=focus_node_indices)

        # Value (critic)
        value = self._critic(batch).squeeze(0)

        # Action log-prob
        logp = torch.tensor(0.0, device=self.device)
        entropy = torch.tensor(0.0, device=self.device)

        if action["type"] == "stop":
            stop_logit = out["stop_logits"].squeeze(-1).squeeze(0)
            logp = _log_prob_bernoulli_from_logit(stop_logit, 1)
            # Bernoulli entropy approximation
            p = torch.sigmoid(stop_logit).clamp(1e-6, 1 - 1e-6)
            entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))

        elif action["type"] == "add_atom":
            add_logits = out["add_node_logits"].squeeze(0)
            atom_idx = _atom_type_to_index(action["atom_type"])
            logp_atom = F.log_softmax(add_logits, dim=-1)[atom_idx]
            logp = logp + logp_atom
            entropy = entropy + (-torch.sum(torch.softmax(add_logits, dim=-1) * F.log_softmax(add_logits, dim=-1)))

            # Optional parent bond supervision at add_atom step (except for the first atom)
            bond_type = action.get("bond_type", None)
            if bond_type is not None and out.get("add_bond_logits", None) is not None:
                bond_logits = out["add_bond_logits"].squeeze(0)
                bt_idx = _bond_type_to_index(bond_type)
                logp = logp + F.log_softmax(bond_logits, dim=-1)[bt_idx]
                entropy = entropy + (-torch.sum(torch.softmax(bond_logits, dim=-1) * F.log_softmax(bond_logits, dim=-1)))

        elif action["type"] == "add_bond":
            target = int(action["target_node_idx"])
            bond_type = action.get("bond_type", Chem.rdchem.BondType.SINGLE)

            edge_logits = out.get("edge_selection_logits", None)
            bond_logits_all = out.get("bond_type_logits", None)
            if edge_logits is None or bond_logits_all is None:
                # If the model can't score this action, make it extremely unlikely.
                logp = torch.tensor(-20.0, device=self.device)
                entropy = torch.tensor(0.0, device=self.device)
            else:
                edge_scores = edge_logits.squeeze(-1)  # [num_nodes]
                logp_edge = F.log_softmax(edge_scores, dim=0)[target]
                logp = logp + logp_edge

                bond_logits = bond_logits_all[target]
                bt_idx = _bond_type_to_index(bond_type)
                logp = logp + F.log_softmax(bond_logits, dim=-1)[bt_idx]

                entropy = entropy + (-torch.sum(torch.softmax(edge_scores, dim=0) * F.log_softmax(edge_scores, dim=0)))
                entropy = entropy + (-torch.sum(torch.softmax(bond_logits, dim=-1) * F.log_softmax(bond_logits, dim=-1)))

        else:
            # Unknown action
            logp = torch.tensor(-20.0, device=self.device)

        return value, logp, entropy

    def _evaluate_smiles(self, smiles: str, score: float) -> List[StepRecord]:
        """
        Convert a SMILES into a supervised trajectory and evaluate old logprobs and rtgs.
        """
        traj = MoleculeTrajectory(smiles).generate_steps()
        # traj is list[(Data, action_dict)]
        n = len(traj)
        rtg = _rtg_for_trajectory(score, n, self.discount_factor)

        records: List[StepRecord] = []
        with torch.no_grad():
            for i, (data, action) in enumerate(traj):
                v, logp, _ = self._evaluate_step(data, action)
                records.append(
                    StepRecord(
                        data=data,
                        action=action,
                        rtg=float(rtg[i].item()),
                        old_logprob=float(logp.detach().cpu().item()),
                    )
                )
        return records

    def update(self, smiles: List[str]) -> None:
        self._actor.model.train()
        self._critic.train()

        # Score
        try:
            score_summary = self._scoring_function.get_final_score_for_step(smiles, self.step)
        except TypeError as inst:
            print(inst, flush=True)
            score_summary = FinalSummary(np.zeros((len(smiles),), dtype=np.float32), smiles, [], [])

        score_summary = deepcopy(score_summary)
        score = self._diversity_filter.update_score(score_summary, self.step)

        # Replay buffer sampling (mirrors PPO.py)
        sample_smiles, sample_score = self._replay_buffer(score_summary.scored_smiles, score)

        # Build step records (teacher forcing on trajectories)
        step_records: List[StepRecord] = []
        for smi, sc in zip(sample_smiles, sample_score):
            try:
                step_records.extend(self._evaluate_smiles(smi, float(sc)))
            except Exception:
                # Skip problematic molecules/trajectories
                continue

        if not step_records:
            self.step += 1
            return

        # Compute advantages (flattened over steps)
        # We recompute values under current critic each PPO epoch, but we need initial values for advantages.
        with torch.no_grad():
            values0 = []
            for rec in step_records:
                batch = Batch.from_data_list([rec.data]).to(self.device)
                values0.append(self._critic(batch).squeeze(0).detach())
            values0 = torch.stack(values0, dim=0)  # [N]

        rtgs = torch.tensor([rec.rtg for rec in step_records], device=self.device, dtype=torch.float32)
        adv = rtgs - values0
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        old_log_probs = torch.tensor([rec.old_logprob for rec in step_records], device=self.device, dtype=torch.float32)

        n_batch = len(step_records)
        mb_size = n_batch // self.n_minibatches if n_batch > self.n_minibatches else n_batch

        actor_losses = []
        critic_losses = []
        entropies = []

        for _ in range(self.n_updates_per_iteration):
            perm = torch.randperm(n_batch, device=self.device)
            for start in range(0, n_batch, mb_size):
                idxs = perm[start : start + mb_size]
                # Current minibatch computation (loop for safety)
                curr_logp = []
                curr_v = []
                curr_ent = []
                for j in idxs.tolist():
                    rec = step_records[j]
                    v, logp, ent = self._evaluate_step(rec.data, rec.action)
                    curr_logp.append(logp)
                    curr_v.append(v)
                    curr_ent.append(ent)
                curr_logp = torch.stack(curr_logp, dim=0)
                curr_v = torch.stack(curr_v, dim=0)
                curr_ent = torch.stack(curr_ent, dim=0)

                ratios = torch.exp(curr_logp - old_log_probs[idxs]).nan_to_num()
                surr1 = ratios * adv[idxs]
                surr2 = adv[idxs] * torch.clamp(ratios, 1 - self.clip, 1 + self.clip)
                actor_loss = -torch.minimum(surr1, surr2).mean()

                if self.use_entropy_bonus:
                    actor_loss = actor_loss - self.entropy_coeff * curr_ent.mean()

                critic_loss = 0.5 * F.mse_loss(curr_v, rtgs[idxs])

                self._actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self._actor.get_network_parameters(), self.max_grad_norm, error_if_nonfinite=False)
                self._actor_optimizer.step()

                self._critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._critic.parameters(), self.max_grad_norm, error_if_nonfinite=False)
                self._critic_optimizer.step()

                actor_losses.append(float(actor_loss.detach().cpu().item()))
                critic_losses.append(float(critic_loss.detach().cpu().item()))
                entropies.append(float(curr_ent.mean().detach().cpu().item()))

        # Logging similar to PPO.py
        valid_fraction = fraction_valid_smiles(self.smiles)
        mean_score = float(np.mean(score)) if len(score) else 0.0

        timestep_report = (
            f"\n Step {self.step} Fraction valid SMILES: {valid_fraction:4.1f} Score: {mean_score:.4f}\n"
            f"Critic loss: {np.mean(critic_losses):.6f}\n"
            f"Actor loss: {np.mean(actor_losses):.6f}\n"
            f"Policy entropy (approx): {np.mean(entropies):.6f}\n"
        )
        self._logger.log_message(timestep_report)

        if self.step % 500 == 0:
            # Save actor weights in the same place the logger expects
            self._logger.save_intermediate_state(self._actor, self._diversity_filter)

        self.step += 1

    def log_out(self) -> None:
        self._logger.save_final_state(self._actor, self._diversity_filter)


