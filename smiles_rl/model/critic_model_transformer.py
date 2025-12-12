"""
Transformer-based Critic for value estimation in RL.

Solves the state representation mismatch issue by using
proper sequence pooling to get a single value per position.

The key insight from the paper is that the "state" should summarize
the full generation history. In RNNs, this is the hidden state.
For Transformers, we need to aggregate across the sequence.
"""

import torch
import torch.nn as tnn
import torch.nn.functional as F
from typing import Optional

from .vocabulary import Vocabulary
from .smiles_tokenizer import SMILESTokenizer


class TransformerCriticNetwork(tnn.Module):
    """
    Transformer network for value function estimation.
    
    Uses the final transformer hidden state at each position
    (which has attended to all previous positions due to causal masking)
    to estimate the value.
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int = 256, 
        n_layers: int = 4, 
        n_heads: int = 4, 
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = tnn.Embedding(vocab_size, d_model)
        self.pos_embedding = tnn.Embedding(max_seq_len, d_model)
        
        encoder_layer = tnn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = tnn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Value head: maps hidden state to scalar
        self.value_head = tnn.Sequential(
            tnn.Linear(d_model, d_model),
            tnn.ReLU(),
            tnn.Linear(d_model, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                tnn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Token indices [batch, seq_len]
            mask: Optional attention mask
            
        Returns:
            values: Value estimates [batch, seq_len]
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Embeddings
        tok_emb = self.embedding(x)  # [batch, seq_len, d_model]
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos)  # [1, seq_len, d_model]
        
        h = tok_emb + pos_emb  # [batch, seq_len, d_model]
        
        # Causal mask for autoregressive value estimation
        # Each position can only attend to itself and previous positions
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        
        # Transformer forward
        h = self.transformer(h, mask=causal_mask)  # [batch, seq_len, d_model]
        
        # Value at each position
        # The hidden state at position t contains information from tokens 0...t
        # This is analogous to the RNN hidden state at step t
        values = self.value_head(h).squeeze(-1)  # [batch, seq_len]
        
        return values


class CriticModelTransformer:
    """
    Transformer-based Critic for value estimation.
    
    Unlike the RNN critic, this properly handles transformer sequences
    by using causal attention to build up state representations.
    
    The value at position t represents V(s_t) where s_t is the
    "state" after generating tokens 0...t, analogous to h_t in RNNs.
    """
    
    def __init__(
        self,
        vocabulary: Vocabulary,
        tokenizer,
        network_params: Optional[dict] = None,
        max_sequence_length: int = 256,
        no_cuda: bool = False,
    ):
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        
        if network_params is None:
            network_params = {}
        
        # Create network
        self.network = TransformerCriticNetwork(
            vocab_size=len(vocabulary),
            d_model=network_params.get("d_model", 256),
            n_layers=network_params.get("n_layers", 4),
            n_heads=network_params.get("n_heads", 4),
            dropout=network_params.get("dropout", 0.1),
            max_seq_len=max_sequence_length,
        )
        
        if torch.cuda.is_available() and not no_cuda:
            self.network.cuda()
    
    @classmethod
    def load_from_file(cls, file_path: str, sampling_mode: bool = False):
        """
        Load critic from a saved checkpoint.
        
        Can initialize from an RNN prior (uses vocab only) or
        a saved transformer critic checkpoint.
        """
        if torch.cuda.is_available():
            save_dict = torch.load(file_path)
        else:
            save_dict = torch.load(file_path, map_location='cpu')
        
        # Get vocabulary
        vocabulary = save_dict["vocabulary"]
        tokenizer = save_dict.get("tokenizer", SMILESTokenizer())
        max_seq_len = save_dict.get("max_sequence_length", 256)
        
        # Use transformer params if available, else defaults
        network_params = save_dict.get("network_params", {})
        if "layer_size" in network_params:
            # Converting from RNN params
            network_params = {
                "d_model": network_params.get("layer_size", 256),
                "n_layers": 4,
                "n_heads": 4,
                "dropout": 0.1,
            }
        
        model = cls(
            vocabulary=vocabulary,
            tokenizer=tokenizer,
            network_params=network_params,
            max_sequence_length=max_seq_len,
        )
        
        # Load weights if this is a transformer critic checkpoint
        if "critic_network" in save_dict:
            model.network.load_state_dict(save_dict["critic_network"])
        
        if sampling_mode:
            model.set_mode("inference")
        else:
            model.set_mode("training")
        
        return model
    
    def set_mode(self, mode: str):
        if mode == "training":
            self.network.train()
        elif mode == "inference":
            self.network.eval()
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def values(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Get value estimates for each position in the sequence.
        
        Args:
            sequences: Token sequences [batch, seq_len]
            
        Returns:
            values: Value estimates [batch, seq_len-1]
                    (excluding last token for consistency with actor log probs)
        """
        # Process through transformer (excludes last token for consistency)
        values = self.network(sequences[:, :-1])
        return values
    
    def get_vocabulary(self):
        return self.vocabulary
    
    def get_network_parameters(self):
        return self.network.parameters()
    
    def save(self, file: str):
        """Save critic state."""
        save_dict = {
            "vocabulary": self.vocabulary,
            "tokenizer": self.tokenizer,
            "max_sequence_length": self.max_sequence_length,
            "critic_network": self.network.state_dict(),
            "network_params": {
                "d_model": self.network.d_model,
            },
        }
        torch.save(save_dict, file)
    
    def save_to_file(self, path: str):
        self.save(path)
    
    def load_state_dict(self, state_dict: dict):
        self.network.load_state_dict(state_dict)
    
    def state_dict(self):
        return self.network.state_dict()


class CriticModelGPT2:
    """
    Critic that works with GPT-2/MolGPT tokenizers (HuggingFace).
    
    Shares vocabulary with the GPT-2 actor, solving the
    vocabulary mismatch issue.
    """
    
    def __init__(
        self,
        tokenizer,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        no_cuda: bool = False,
    ):
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        
        self.network = TransformerCriticNetwork(
            vocab_size=self.vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        
        if torch.cuda.is_available() and not no_cuda:
            self.network.cuda()
        
        print(f"✅ CriticModelGPT2: vocab={self.vocab_size}, d_model={d_model}, layers={n_layers}")
    
    @classmethod
    def from_actor(cls, actor, d_model: int = 256, n_layers: int = 4):
        """
        Create a critic that matches an ActorModelGPT2's vocabulary.
        """
        return cls(
            tokenizer=actor.tokenizer,
            d_model=d_model,
            n_layers=n_layers,
            max_seq_len=actor.max_length,
        )
    
    def values(self, sequences: torch.Tensor) -> torch.Tensor:
        """Get values for sequence positions."""
        return self.network(sequences[:, :-1])
    
    def get_vocabulary(self):
        """Return vocab wrapper compatible with equality checks."""
        class _VocabWrapper:
            def __init__(self, tokenizer):
                self._tokens = list(tokenizer.get_vocab().keys())
            def __len__(self):
                return len(self._tokens)
            def __eq__(self, other):
                # For GPT2 actors, just check vocab size
                return len(self) == len(other)
        return _VocabWrapper(self.tokenizer)
    
    def get_network_parameters(self):
        return self.network.parameters()
    
    def set_mode(self, mode: str):
        if mode == "training":
            self.network.train()
        else:
            self.network.eval()
