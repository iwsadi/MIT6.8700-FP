"""
Transformer-based policy/value model for SMILES using ChemGPT checkpoints.
"""
from typing import Optional

import torch
import torch.nn as tnn
import torch.nn.functional as tnnf


try:
    from transformers import AutoModelForCausalLM
except ImportError as exc:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None
    _transformers_import_error = exc
else:
    _transformers_import_error = None


class ChemGPTTransformer(tnn.Module):
    """
    Wraps a pretrained ChemGPT causal LM to expose the same interface as RNN:
    returns logits and an optional cached state (past_key_values).
    Single output head - suitable for ActorModel or CriticModel.
    """

    def __init__(
        self,
        model_name_or_path: str,
        pad_token_id: Optional[int] = None,
        fine_tune: bool = True,
        layer_normalization: bool = False,
        use_cache: bool = True,
        output_size: Optional[int] = None,
    ):
        super().__init__()
        if AutoModelForCausalLM is None:
            raise ImportError(
                "transformers is required for ChemGPT backend"
            ) from _transformers_import_error

        self._model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self._model_name_or_path = model_name_or_path

        hidden_size = self._model.config.hidden_size
        vocab_size = output_size if output_size is not None else self._model.config.vocab_size

        # Output head - reuse LM head if output_size matches, else create new
        if output_size is None or output_size == self._model.config.vocab_size:
            lm_head = self._model.get_output_embeddings()
            self._linear = lm_head if lm_head is not None else tnn.Linear(hidden_size, vocab_size)
        else:
            self._linear = tnn.Linear(hidden_size, vocab_size)

        self._layer_size = hidden_size
        self._pad_token_id = pad_token_id if pad_token_id is not None else self._model.config.pad_token_id
        self._layer_normalization = layer_normalization
        self._use_cache = use_cache
        self._fine_tune = fine_tune

        if not fine_tune:
            for param in self._model.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_vector: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ):
        """
        Performs a forward pass on the transformer.
        :param input_vector: Tensor (batch_size, seq_len) with token ids.
        :param hidden_state: Transformer cache (past_key_values) or None.
        :return: (logits, past_key_values)
        """
        batch_size, curr_len = input_vector.shape
        
        # Calculate past sequence length from KV cache
        past_len = 0
        if hidden_state is not None:
            # past_key_values is tuple of (key, value) for each layer
            # key shape: [batch, num_heads, seq_len, head_dim]
            past_len = hidden_state[0][0].shape[2]
        
        # Build attention mask for the full sequence (past + current)
        attention_mask = None
        if self._pad_token_id is not None:
            # Current tokens mask
            curr_mask = (input_vector != self._pad_token_id).long()
            if past_len > 0:
                # Prepend ones for past tokens (they're always valid, non-padded)
                past_mask = torch.ones(batch_size, past_len, dtype=torch.long, device=input_vector.device)
                attention_mask = torch.cat([past_mask, curr_mask], dim=1)
            else:
                attention_mask = curr_mask

        # Safety: clamp input_ids to valid vocabulary range to prevent CUDA index errors
        # This handles cases where the model generates invalid token indices during collapse
        vocab_size = self._model.config.vocab_size
        input_vector = input_vector.clamp(0, vocab_size - 1)
        
        outputs = self._model(
            input_ids=input_vector,
            attention_mask=attention_mask,
            past_key_values=hidden_state,
            use_cache=self._use_cache,
            output_hidden_states=True,
        )

        hidden = outputs.hidden_states[-1]
        if self._layer_normalization:
            hidden = tnnf.layer_norm(hidden, hidden.size()[1:])

        logits = self._linear(hidden)

        return logits, outputs.past_key_values

    def get_params(self):
        return {
            "backend": "transformer",
            "model_name_or_path": self._model_name_or_path,
            "pad_token_id": self._pad_token_id,
            "layer_normalization": self._layer_normalization,
            "use_cache": self._use_cache,
            "fine_tune": self._fine_tune,
        }

    def get_backbone_parameters(self):
        """Returns parameters of the pretrained backbone (transformer model), excluding LM head."""
        # Get all parameter ids from the linear head
        head_param_ids = set(id(p) for p in self._linear.parameters())
        # Return backbone params that are NOT in the head (to avoid duplicates)
        return (p for p in self._model.parameters() if id(p) not in head_param_ids)

    def get_head_parameters(self):
        """Returns parameters of the output head (linear layer)."""
        return self._linear.parameters()

    def get_parameter_groups(self, lr_backbone: float, lr_head: float):
        """
        Returns parameter groups with different learning rates for backbone and head.
        
        Args:
            lr_backbone: Learning rate for the pretrained backbone
            lr_head: Learning rate for the output head
            
        Returns:
            List of parameter group dicts suitable for torch.optim.Adam
        """
        backbone_params = list(self.get_backbone_parameters())
        head_params = list(self.get_head_parameters())
        
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone})
        if head_params:
            groups.append({"params": head_params, "lr": lr_head})
        return groups

    def freeze_backbone(self):
        """Freeze backbone parameters, only allow head to be trained."""
        for param in self.get_backbone_parameters():
            param.requires_grad = False
        self._fine_tune = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters to allow fine-tuning."""
        for param in self._model.parameters():
            param.requires_grad = True
        self._fine_tune = True


class ChemGPTTransformerShared(tnn.Module):
    """
    Wraps a pretrained ChemGPT causal LM to expose the same interface as RNNShared:
    returns policy logits, critic logits, and an optional cached state (past_key_values).
    """

    def __init__(
        self,
        model_name_or_path: str,
        pad_token_id: Optional[int],
        fine_tune: bool = True,
        layer_normalization: bool = False,
        use_cache: bool = True,
    ):
        super().__init__()
        if AutoModelForCausalLM is None:
            raise ImportError(
                "transformers is required for ChemGPT backend"
            ) from _transformers_import_error

        self._model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

        # Reuse the LM head for policy to stay aligned with the checkpoint.
        policy_head = self._model.get_output_embeddings()
        hidden_size = self._model.config.hidden_size
        self._policy_head = policy_head or tnn.Linear(
            hidden_size, self._model.config.vocab_size
        )

        # Separate critic head to avoid interfering with pretrained logits.
        self._critic_head = tnn.Linear(hidden_size, self._model.config.vocab_size)

        self._pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else self._model.config.pad_token_id
        )
        self._layer_normalization = layer_normalization
        self._use_cache = use_cache

        if not fine_tune:
            for param in self._model.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_vector: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ):
        """
        Performs a forward pass on the transformer.
        :param input_vector: Tensor (batch_size, seq_len) with token ids.
        :param hidden_state: Transformer cache (past_key_values) or None.
        """
        attention_mask = None
        if self._pad_token_id is not None:
            attention_mask = (input_vector != self._pad_token_id).long()

        outputs = self._model(
            input_ids=input_vector,
            attention_mask=attention_mask,
            past_key_values=hidden_state,
            use_cache=self._use_cache,
        )

        hidden = outputs.last_hidden_state
        if self._layer_normalization:
            hidden = tnnf.layer_norm(hidden, hidden.size()[1:])

        policy_logits = self._policy_head(hidden)
        critic_logits = self._critic_head(hidden)

        return policy_logits, critic_logits, outputs.past_key_values

    def get_params(self):
        return {
            "backend": "transformer",
            "model_name_or_path": getattr(self._model, "name_or_path", ""),
            "pad_token_id": self._pad_token_id,
            "layer_normalization": self._layer_normalization,
            "use_cache": self._use_cache,
        }

    def get_backbone_parameters(self):
        """Returns parameters of the pretrained backbone (transformer model), excluding heads."""
        from itertools import chain
        # Get all parameter ids from the heads
        head_param_ids = set(id(p) for p in chain(
            self._policy_head.parameters(), 
            self._critic_head.parameters()
        ))
        # Return backbone params that are NOT in the heads (to avoid duplicates)
        return (p for p in self._model.parameters() if id(p) not in head_param_ids)

    def get_head_parameters(self):
        """Returns parameters of the output heads (policy and critic)."""
        from itertools import chain
        return chain(self._policy_head.parameters(), self._critic_head.parameters())

    def get_parameter_groups(self, lr_backbone: float, lr_head: float):
        """
        Returns parameter groups with different learning rates for backbone and head.
        
        Args:
            lr_backbone: Learning rate for the pretrained backbone
            lr_head: Learning rate for the output heads
            
        Returns:
            List of parameter group dicts suitable for torch.optim.Adam
        """
        backbone_params = list(self.get_backbone_parameters())
        head_params = list(self.get_head_parameters())
        
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone})
        if head_params:
            groups.append({"params": head_params, "lr": lr_head})
        return groups

    def freeze_backbone(self):
        """Freeze backbone parameters, only allow heads to be trained."""
        for param in self.get_backbone_parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters to allow fine-tuning."""
        for param in self._model.parameters():
            param.requires_grad = True
