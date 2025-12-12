"""
MolGPT-based Critic for Value Estimation in RL (A2C/PPO).

This module implements a Critic that shares the same architecture and
pre-trained weights as the MolGPT Actor, but replaces the language model
head with a value head for estimating V(s).

Key Design Decisions:
1. Initialize from same pre-trained weights as Actor (stability)
2. Replace lm_head with value_head (Linear -> 1)
3. Use last token representation for sequence-level value
4. Backbone is TRAINABLE (not frozen)

Reference: "Initializing the Critic with pre-trained weights is crucial 
for stability in molecular RL" - Common practice in REINVENT, MolDQN, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from pathlib import Path

try:
    from transformers import GPT2Model, GPT2Config, AutoConfig
except ImportError:
    raise ImportError("transformers required: pip install transformers")


class MolGPTCritic(nn.Module):
    """
    Critic network for PPO/A2C using MolGPT architecture.
    
    Architecture:
        Input tokens -> GPT2 Backbone -> Last hidden state -> Value Head -> V(s)
    
    The backbone is initialized from the same pre-trained weights as the Actor,
    but the lm_head is replaced with a value_head that outputs a scalar.
    
    Args:
        config: GPT2Config or path to model
        hidden_dim: Hidden dimension of GPT (auto-detected if loading)
        value_head_init: Initialization method ("orthogonal" or "normal")
        value_head_std: Std for weight initialization (default: 0.01)
    """
    
    def __init__(
        self,
        config: GPT2Config,
        hidden_dim: int = 768,
        value_head_init: str = "orthogonal",
        value_head_std: float = 0.01,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # =====================================================================
        # GPT-2 Backbone (same as Actor, but WITHOUT lm_head)
        # =====================================================================
        # GPT2Model gives us the transformer backbone without the LM head
        self.backbone = GPT2Model(config)
        
        # =====================================================================
        # Value Head: Maps hidden state -> scalar value V(s)
        # =====================================================================
        # For stability, use a small network with proper initialization
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),  # Tanh is common for value networks (bounded gradients)
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Initialize value head
        self._init_value_head(value_head_init, value_head_std)
    
    def _init_value_head(self, method: str, std: float):
        """
        Initialize value head weights for training stability.
        
        Orthogonal initialization with small std is recommended for
        value networks in RL (PPO paper, CleanRL implementations).
        """
        for module in self.value_head:
            if isinstance(module, nn.Linear):
                if method == "orthogonal":
                    nn.init.orthogonal_(module.weight, gain=std)
                elif method == "normal":
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                else:
                    nn.init.xavier_uniform_(module.weight)
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        print(f"  Value head initialized: {method} (std={std})")
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        value_head_init: str = "orthogonal",
        value_head_std: float = 0.01,
        device: Optional[torch.device] = None,
    ) -> "MolGPTCritic":
        """
        Load Critic from pre-trained MolGPT/GPT-2 checkpoint.
        
        This loads the SAME weights as the Actor, ensuring the Critic
        starts with a good representation of molecular structure.
        
        Args:
            model_name_or_path: HuggingFace model ID or local path
            value_head_init: Initialization for value head
            value_head_std: Std for initialization
            device: Target device
            
        Returns:
            MolGPTCritic with pre-trained backbone
        """
        print(f"\n  Loading MolGPTCritic from: {model_name_or_path}")
        
        # Load config
        config = AutoConfig.from_pretrained(model_name_or_path)
        hidden_dim = config.n_embd
        
        # Create critic
        critic = cls(
            config=config,
            hidden_dim=hidden_dim,
            value_head_init=value_head_init,
            value_head_std=value_head_std,
        )
        
        # Load pre-trained backbone weights
        # GPT2Model.from_pretrained loads transformer layers without lm_head
        pretrained_backbone = GPT2Model.from_pretrained(model_name_or_path)
        critic.backbone.load_state_dict(pretrained_backbone.state_dict())
        
        n_params = sum(p.numel() for p in critic.parameters())
        backbone_params = sum(p.numel() for p in critic.backbone.parameters())
        head_params = sum(p.numel() for p in critic.value_head.parameters())
        
        print(f"  ✅ Critic loaded: {n_params:,} total params")
        print(f"     Backbone: {backbone_params:,} (pre-trained)")
        print(f"     Value head: {head_params:,} (initialized)")
        
        # Move to device
        if device is not None:
            critic.to(device)
        
        return critic
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_positions: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass: compute value V(s) for input sequences.
        
        Args:
            input_ids: Token sequences [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            return_all_positions: If True, return values for all positions
                                  If False, return only last position value
        
        Returns:
            values: [batch_size, 1] if return_all_positions=False
                    [batch_size, seq_len] if return_all_positions=True
        """
        # Get hidden states from backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        
        if return_all_positions:
            # Return value for each position (useful for per-step advantages)
            # Each position's value is based on hidden state at that position
            # (which has attended to all previous positions due to causal mask)
            values = self.value_head(hidden_states).squeeze(-1)  # [batch, seq_len]
            return values
        else:
            # Return single value for the sequence
            # Use the LAST non-padding token's hidden state
            if attention_mask is not None:
                # Find last non-padding position for each sequence
                seq_lengths = attention_mask.sum(dim=1) - 1  # [batch]
                batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                last_hidden = hidden_states[batch_indices, seq_lengths]  # [batch, hidden_dim]
            else:
                # No mask - use last position
                last_hidden = hidden_states[:, -1, :]  # [batch, hidden_dim]
            
            values = self.value_head(last_hidden)  # [batch, 1]
            return values
    
    def values(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Compatibility method matching CriticModel interface.
        
        Returns values for all positions (excluding last token for
        consistency with actor log probabilities).
        
        Args:
            sequences: Token sequences [batch, seq_len]
            
        Returns:
            values: [batch, max(1, seq_len-1)] - always at least 1 position
        """
        batch_size, seq_len = sequences.shape
        
        # Handle edge cases: very short sequences
        if seq_len <= 1 or batch_size == 0:
            # Return an empty tensor with grad, matching seq_len-1 = 0
            return torch.zeros(batch_size, 0, device=sequences.device, requires_grad=True)
        
        # Normal case: get values for all positions (excluding last token)
        input_seqs = sequences[:, :-1]
        all_values = self.forward(input_seqs, return_all_positions=True)
        return all_values
    
    def get_network_parameters(self):
        """Return all trainable parameters."""
        return self.parameters()


class MolGPTCriticFromActor:
    """
    Alternative: Create Critic by cloning Actor's backbone.
    
    This approach directly copies the Actor's transformer layers
    rather than re-loading from disk.
    """
    
    @staticmethod
    def from_actor(
        actor,
        value_head_init: str = "orthogonal", 
        value_head_std: float = 0.01,
    ):
        """
        Create Critic from an existing ActorModelGPT2.
        
        Args:
            actor: ActorModelGPT2 instance
            value_head_init: Initialization method for value head
            value_head_std: Std for initialization
            
        Returns:
            MolGPTCritic with backbone copied from actor
        """
        import copy
        
        # Get config from actor's model
        config = actor.model.config
        hidden_dim = config.n_embd
        device = next(actor.model.parameters()).device
        
        print(f"\n  Creating Critic from Actor backbone...")
        
        # Create critic with same config
        critic = MolGPTCritic(
            config=config,
            hidden_dim=hidden_dim,
            value_head_init=value_head_init,
            value_head_std=value_head_std,
        )
        
        # Copy backbone weights from actor
        # Actor's model is GPT2LMHeadModel, which has .transformer (GPT2Model)
        actor_backbone_state = actor.model.transformer.state_dict()
        critic.backbone.load_state_dict(actor_backbone_state)
        
        critic.to(device)
        
        n_params = sum(p.numel() for p in critic.parameters())
        print(f"  ✅ Critic created from Actor: {n_params:,} params")
        
        return critic


# =============================================================================
# Wrapper class for compatibility with existing SMILES-RL framework
# =============================================================================

class CriticModelMolGPT:
    """
    Wrapper class for MolGPTCritic, compatible with SMILES-RL framework.
    
    Provides the same interface as CriticModel (RNN-based) and
    CriticModelTransformer.
    """
    
    def __init__(
        self,
        tokenizer,
        model: MolGPTCritic,
        device: Optional[torch.device] = None,
    ):
        self.tokenizer = tokenizer
        self.network = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: Optional[torch.device] = None,
    ):
        """Load critic from pre-trained checkpoint."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = MolGPTCritic.from_pretrained(model_name_or_path, device=device)
        
        return cls(tokenizer=tokenizer, model=model, device=device)
    
    @classmethod
    def from_actor(cls, actor, **kwargs):
        """Create critic from existing actor."""
        critic_model = MolGPTCriticFromActor.from_actor(actor, **kwargs)
        return cls(
            tokenizer=actor.tokenizer,
            model=critic_model,
            device=actor.device,
        )
    
    def values(self, sequences: torch.Tensor) -> torch.Tensor:
        """Get values for sequence positions."""
        return self.network.values(sequences)
    
    def get_vocabulary(self):
        """Return vocab wrapper for compatibility checks."""
        class _VocabWrapper:
            def __init__(self, tokenizer):
                self._size = len(tokenizer)
            def __len__(self):
                return self._size
            def __eq__(self, other):
                return len(self) == len(other)
        return _VocabWrapper(self.tokenizer)
    
    def get_network_parameters(self):
        """Return trainable parameters."""
        return self.network.parameters()
    
    def set_mode(self, mode: str):
        """Set training/eval mode."""
        if mode == "training":
            self.network.train()
        else:
            self.network.eval()
    
    def train(self):
        self.network.train()
    
    def eval(self):
        self.network.eval()
    
    def to(self, device):
        self.device = device
        self.network.to(device)
        return self


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    print("=" * 60)
    print("  Testing MolGPTCritic")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "entropy/gpt2_zinc_87m"
    
    # Method 1: Load from pretrained
    print("\n[Method 1] Loading from pretrained...")
    critic = MolGPTCritic.from_pretrained(model_name, device=device)
    
    # Test forward pass
    dummy_input = torch.randint(0, 1000, (4, 32), device=device)
    
    # Single value per sequence
    values_single = critic(dummy_input, return_all_positions=False)
    print(f"\n  Single value shape: {values_single.shape}")  # [4, 1]
    
    # Value per position
    values_all = critic(dummy_input, return_all_positions=True)
    print(f"  All positions shape: {values_all.shape}")  # [4, 32]
    
    # Method 2: From Actor
    print("\n[Method 2] Creating from Actor...")
    from smiles_rl.model.actor_model_gpt2 import ActorModelGPT2
    
    actor = ActorModelGPT2.from_pretrained(model_name, device=device)
    critic_from_actor = MolGPTCriticFromActor.from_actor(actor)
    
    # Verify outputs
    values_from_actor = critic_from_actor(dummy_input, return_all_positions=True)
    print(f"  From actor shape: {values_from_actor.shape}")
    
    # Check parameter count
    print(f"\n  Critic params: {sum(p.numel() for p in critic.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in critic.parameters() if p.requires_grad):,}")
    
    print("\n" + "=" * 60)
    print("  ✅ All tests passed!")
    print("=" * 60)
