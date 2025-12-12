"""
GPT-2 / MolGPT actor wrapper for SMILES-RL.

Supports:
- HuggingFace MolGPT: jonghyunlee/MolGPT_pretrained-by-ZINC15
- Local saved models from download_molgpt.py
- entropy/gpt2_zinc_87m

Provides proper sampling with temperature for RL training.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from transformers import GPT2LMHeadModel, AutoTokenizer, PreTrainedTokenizerFast
except ImportError as e:
    raise ImportError("Please install transformers: pip install transformers") from e


class ActorModelGPT2:
    """
    GPT-2 based actor for molecular generation.
    
    Compatible with HuggingFace MolGPT models and SMILES-RL framework.
    """
    
    def __init__(
        self, 
        model: GPT2LMHeadModel, 
        tokenizer, 
        max_length: int = 128, 
        device: Optional[torch.device] = None,
        temperature: float = 1.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.temperature = temperature
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Cache special token IDs
        self._bos_id = tokenizer.bos_token_id
        self._eos_id = tokenizer.eos_token_id
        self._pad_id = tokenizer.pad_token_id

    @classmethod
    def from_pretrained(
        cls, 
        model_name_or_path: str, 
        max_length: int = 128, 
        device: Optional[torch.device] = None,
        temperature: float = 1.0,
    ):
        """
        Load from HuggingFace hub or local directory.
        
        Args:
            model_name_or_path: HuggingFace model name or local path
            max_length: Maximum sequence length
            device: Torch device
            temperature: Sampling temperature
        """
        from transformers import GPT2Tokenizer
        
        path = Path(model_name_or_path)
        
        # Try loading tokenizer with multiple methods for compatibility
        tokenizer = None
        tokenizer_methods = [
            ("AutoTokenizer", lambda: AutoTokenizer.from_pretrained(model_name_or_path)),
            ("AutoTokenizer (trust_remote)", lambda: AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)),
            ("AutoTokenizer (slow)", lambda: AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)),
            ("GPT2Tokenizer", lambda: GPT2Tokenizer.from_pretrained(model_name_or_path)),
            ("PreTrainedTokenizerFast", lambda: PreTrainedTokenizerFast.from_pretrained(model_name_or_path)),
        ]
        
        for method_name, loader in tokenizer_methods:
            try:
                tokenizer = loader()
                print(f"  Tokenizer loaded via {method_name}")
                break
            except Exception as e:
                continue
        
        # Load model first to get vocab_size
        model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        
        if tokenizer is None:
            # Last resort: create a SMILES tokenizer matching model's vocab_size
            print(f"  ⚠️ Could not load tokenizer, creating fallback (vocab={model.config.vocab_size})")
            tokenizer = cls._create_smiles_tokenizer(model.config.vocab_size)
        
        # Ensure special tokens are set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<pad>"
        if tokenizer.bos_token is None:
            # Try to find a suitable BOS token
            if hasattr(tokenizer, 'get_vocab'):
                vocab = tokenizer.get_vocab()
                for candidate in ["<s>", "[CLS]", "<bos>", "<start>"]:
                    if candidate in vocab:
                        tokenizer.bos_token = candidate
                        break
            if tokenizer.bos_token is None:
                tokenizer.add_special_tokens({"bos_token": "<bos>"})
                model.resize_token_embeddings(len(tokenizer))
        if tokenizer.eos_token is None:
            tokenizer.eos_token = "<eos>"
        
        # Resize embeddings if needed
        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
        
        print(f"✅ Loaded MolGPT: {model.num_parameters():,} params, vocab={len(tokenizer)}")
        
        return cls(model, tokenizer, max_length=max_length, device=device, temperature=temperature)
    
    @staticmethod
    def _create_smiles_tokenizer(vocab_size: int = 2140):
        """Create a SMILES tokenizer matching the model's vocab size."""
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        
        # Build SMILES vocabulary to match model's vocab_size
        # Order matters - must match what model was trained on
        tokens = [
            '<pad>', '<s>', '</s>', '<unk>',
            # Single character atoms
            'C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'P', 'p', 'F', 'I', 'B', 'b',
            # Bonds
            '=', '#', '-', '+', '\\', '/', ':', '.',
            # Structure
            '(', ')', '[', ']',
            # Ring numbers
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '%',
            # Stereo
            '@', '@@',
            # Multi-char atoms
            'Cl', 'Br', 'Si', 'Se', 'se', 'Te', 'te', 'As', 'as',
            'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'Zn', 'Cu', 'Mn', 'Al', 'Sn', 'Pb',
            # Charges
            'H', '+1', '+2', '+3', '-1', '-2', '-3', 'H1', 'H2', 'H3',
            # Aromatics
            'nH', 'oH', 'sH',
        ]
        
        # Fill remaining vocab with extra tokens
        while len(tokens) < vocab_size:
            tokens.append(f'<extra_{len(tokens)}>')
        
        tokens = tokens[:vocab_size]
        vocab = {t: i for i, t in enumerate(tokens)}
        
        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token='<unk>'))
        
        # Create HuggingFace tokenizer
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token='<s>',
            eos_token='</s>',
            pad_token='<pad>',
            unk_token='<unk>',
        )
        
        return hf_tokenizer

    def get_vocabulary(self):
        """Return vocab-like object compatible with SMILES-RL."""
        class _VocabWrapper:
            def __init__(self, tokenizer):
                self._tokens = list(tokenizer.get_vocab().keys())
                self._vocab = tokenizer.get_vocab()
                
            def __len__(self):
                return len(self._tokens)
            
            def __getitem__(self, token):
                return self._vocab.get(token, 0)
            
            def __contains__(self, token):
                return token in self._vocab
                
        return _VocabWrapper(self.tokenizer)
    
    @property
    def vocabulary(self):
        """Alias for get_vocabulary."""
        return self.get_vocabulary()

    def get_network_parameters(self):
        """Return model parameters for optimizer."""
        return self.model.parameters()

    def smiles_to_sequences(self, smiles: List[str]) -> torch.Tensor:
        """
        Convert SMILES strings to token sequences.
        
        The tokenizer is WordLevel, so we need to manually tokenize SMILES
        character-by-character (or using SMILES-specific patterns) before encoding.
        """
        import re
        
        # Clean SMILES before tokenization (remove ByteLevel artifacts)
        cleaned_smiles = []
        for smi in smiles:
            if not smi or len(smi.strip()) == 0:
                continue  # Skip empty SMILES
            # Remove ByteLevel tokenizer artifacts that might be present
            smi = smi.lstrip("Ġ")  # Remove space prefix token
            smi = smi.lstrip("─á")  # Remove BOS artifacts
            smi = smi.strip()
            if len(smi) > 0:
                cleaned_smiles.append(smi)
        
        if len(cleaned_smiles) == 0:
            # Return empty tensor with proper shape (batch=0, seq_len=2 minimum)
            return torch.zeros((0, 2), dtype=torch.long, device=self.device)
        
        # Tokenize SMILES using regex patterns (similar to SMILESTokenizer)
        # This matches how the model was trained
        REGEXPS = {
            "brackets": re.compile(r"(\[[^\]]*\])"),
            "2_ring_nums": re.compile(r"(%\d{2})"),
            "brcl": re.compile(r"(Br|Cl)"),
        }
        REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]
        
        def split_by(data, regexps):
            """Recursively split SMILES using regex patterns."""
            if not regexps:
                return list(data)  # Character-level split
            regexp = REGEXPS[regexps[0]]
            splitted = regexp.split(data)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)  # Keep matched pattern as single token
            return tokens
        
        # Tokenize each SMILES string
        tokenized_smiles = []
        for smi in cleaned_smiles:
            tokens = split_by(smi, REGEXP_ORDER)
            tokenized_smiles.append(tokens)
        
        # Encode tokens to IDs using tokenizer's vocabulary
        # We need to handle tokens that might not be in vocab (map to <unk>)
        encoded_sequences = []
        for tokens in tokenized_smiles:
            token_ids = []
            for token in tokens:
                # Try to encode token, fallback to <unk> if not found
                try:
                    # Use tokenizer's convert_tokens_to_ids
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    if token_id == self.tokenizer.unk_token_id:
                        # Token not in vocab, try to encode as string
                        enc = self.tokenizer.encode(token, add_special_tokens=False)
                        if len(enc) > 0:
                            token_ids.extend(enc)
                        else:
                            token_ids.append(self.tokenizer.unk_token_id)
                    else:
                        token_ids.append(token_id)
                except:
                    token_ids.append(self.tokenizer.unk_token_id)
            
            # Add BOS and EOS tokens
            if self._bos_id is not None:
                token_ids = [self._bos_id] + token_ids
            if self._eos_id is not None:
                token_ids = token_ids + [self._eos_id]
            
            encoded_sequences.append(token_ids)
        
        # Pad sequences to same length
        max_len = min(max(len(seq) for seq in encoded_sequences), self.max_length)
        padded_sequences = []
        for seq in encoded_sequences:
            # Truncate if too long
            if len(seq) > max_len:
                seq = seq[:max_len]
            # Pad with pad_id
            pad_length = max_len - len(seq)
            padded_seq = seq + [self._pad_id] * pad_length
            padded_sequences.append(padded_seq)
        
        seqs = torch.tensor(padded_sequences, dtype=torch.long, device=self.device)
        
        return seqs

    def log_probabilities_action(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Compute log probabilities of actions (tokens) in sequences.
        
        Args:
            sequences: Token sequences (batch, seq_len)
            
        Returns:
            Log probabilities of shape (batch, seq_len - 1)
        """
        with torch.set_grad_enabled(self.model.training):
            outputs = self.model(sequences)
            logits = outputs.logits[:, :-1, :]  # (batch, seq_len-1, vocab)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Gather log probs for actual tokens
            targets = sequences[:, 1:].unsqueeze(-1)  # (batch, seq_len-1, 1)
            action_log_probs = log_probs.gather(-1, targets).squeeze(-1)  # (batch, seq_len-1)
            
        return action_log_probs

    def log_and_probabilities(self, sequences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute full log probabilities and probabilities for sequences.
        
        Returns:
            log_probs: (batch, seq_len, vocab_size)
            probs: (batch, seq_len, vocab_size)
        """
        with torch.no_grad():
            outputs = self.model(sequences)
            logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        return log_probs, probs

    def sample(
        self, 
        batch_size: int = 32,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """
        Sample SMILES sequences autoregressively.
        
        Args:
            batch_size: Number of sequences to generate
            temperature: Sampling temperature (uses self.temperature if None)
            
        Returns:
            sequences: Token IDs (batch, seq_len)
            smiles: Decoded SMILES strings
            log_probs: Log probabilities of actions (batch, seq_len-1)
        """
        temp = temperature if temperature is not None else self.temperature
        self.model.eval()
        
        # Start with BOS token
        sequences = torch.full(
            (batch_size, 1), 
            self._bos_id, 
            device=self.device, 
            dtype=torch.long
        )
        
        # Track which sequences have finished (hit EOS)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # Collect log probabilities
        all_log_probs = []
        
        with torch.no_grad():
            for step in range(self.max_length - 1):
                # Forward pass
                outputs = self.model(sequences)
                next_token_logits = outputs.logits[:, -1, :]  # (batch, vocab)
                
                # Apply temperature
                if temp != 1.0:
                    next_token_logits = next_token_logits / temp
                
                # Sample from distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)  # (batch, 1)
                
                # Compute log probabilities of sampled tokens
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                sampled_log_probs = log_probs.gather(-1, next_tokens).squeeze(-1)  # (batch,)
                all_log_probs.append(sampled_log_probs)
                
                # Replace tokens for finished sequences with PAD
                next_tokens = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_tokens, self._pad_id),
                    next_tokens
                )
                
                # Append to sequences
                sequences = torch.cat([sequences, next_tokens], dim=1)
                
                # Update finished status
                finished = finished | (next_tokens.squeeze(-1) == self._eos_id)
                
                # Early exit if all sequences finished
                if finished.all():
                    break
        
        # Stack log probabilities
        log_probs_tensor = torch.stack(all_log_probs, dim=1)  # (batch, generated_len)
        
        # Decode to SMILES
        # Note: ByteLevel tokenizers sometimes don't properly skip special tokens
        # So we manually filter out BOS/EOS/PAD tokens before decoding
        sequences_to_decode = sequences.clone()
        
        # Remove special tokens (BOS, EOS, PAD) by replacing with pad_id before decoding
        # This ensures they're skipped by skip_special_tokens=True
        if self._bos_id is not None:
            sequences_to_decode[sequences_to_decode == self._bos_id] = self._pad_id
        if self._eos_id is not None:
            sequences_to_decode[sequences_to_decode == self._eos_id] = self._pad_id
        
        smiles = self.tokenizer.batch_decode(sequences_to_decode, skip_special_tokens=True)
        
        # Clean up whitespace and any remaining special token artifacts
        # ByteLevel tokenizers can produce weird characters like "─á" from BOS token
        cleaned_smiles = []
        bos_token_str = self.tokenizer.bos_token if self.tokenizer.bos_token else ""
        eos_token_str = self.tokenizer.eos_token if self.tokenizer.eos_token else ""
        
        for smi in smiles:
            # Remove whitespace
            smi = smi.replace(" ", "")
            # Remove explicit special token strings
            smi = smi.replace("<bos>", "").replace("<eos>", "").replace("<pad>", "")
            smi = smi.replace("<s>", "").replace("</s>", "")
            if bos_token_str:
                smi = smi.replace(bos_token_str, "")
            if eos_token_str:
                smi = smi.replace(eos_token_str, "")
            # Remove byte-level encoding artifacts (common with ByteLevel tokenizers)
            # The "─á" pattern is a common artifact of BOS token byte-level encoding
            smi = smi.lstrip("─á")  # Remove leading weird chars from BOS
            # Remove ByteLevel tokenizer space prefix "Ġ" (common with GPT-2 style tokenizers)
            smi = smi.lstrip("Ġ")  # Remove leading space prefix token
            smi = smi.strip()
            cleaned_smiles.append(smi)
        
        return sequences, cleaned_smiles, log_probs_tensor

    def save(self, file: str):
        """Save model and tokenizer."""
        save_path = Path(file)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as directory (HuggingFace format)
        if save_path.suffix == "":
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
        else:
            # Save as single file (PyTorch format)
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "config": self.model.config.to_dict(),
                "max_length": self.max_length,
                "temperature": self.temperature,
            }, file)
            # Save tokenizer separately
            tokenizer_path = save_path.parent / "tokenizer"
            self.tokenizer.save_pretrained(tokenizer_path)

    def save_to_file(self, path: str):
        """Alias for save."""
        self.save(path)
    
    def train(self):
        """Set model to training mode."""
        self.model.train()
        
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
    
    def to(self, device):
        """Move model to device."""
        self.device = device
        self.model.to(device)
        return self

