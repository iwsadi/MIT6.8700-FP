"""
SMILES Dataset for Supervised Pre-training.

Provides a PyTorch Dataset that tokenizes SMILES strings and returns
(input_seq, target_seq) pairs for next-token prediction training.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
import numpy as np

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from smiles_rl.model.vocabulary import Vocabulary, SMILESTokenizer


class SMILESDataset(Dataset):
    """
    PyTorch Dataset for SMILES strings with next-token prediction.
    
    For each SMILES, returns:
        - input_seq:  [^, t1, t2, ..., tN]     (start token + SMILES tokens)
        - target_seq: [t1, t2, ..., tN, $]     (SMILES tokens + end token)
    
    The target is the input shifted by one position, which is the standard
    setup for autoregressive language modeling.
    """
    
    def __init__(
        self,
        smiles_list: List[str],
        vocabulary: Vocabulary,
        tokenizer: Optional[SMILESTokenizer] = None,
        max_length: int = 100,
    ):
        """
        Initialize the SMILES dataset.
        
        Args:
            smiles_list: List of SMILES strings
            vocabulary: Vocabulary object for encoding tokens
            tokenizer: SMILESTokenizer instance (creates new one if None)
            max_length: Maximum sequence length (SMILES longer than this are filtered)
        """
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer if tokenizer is not None else SMILESTokenizer()
        self.max_length = max_length
        
        # Get special token indices
        self.start_token_idx = vocabulary["^"]
        self.end_token_idx = vocabulary["$"]
        self.pad_token_idx = vocabulary["$"]  # Use end token as padding (index 0)
        
        # Filter and store valid SMILES
        self.smiles_list = []
        self.encoded_sequences = []
        
        n_filtered_length = 0
        n_filtered_vocab = 0
        
        for smi in smiles_list:
            smi = smi.strip()
            if not smi:
                continue
                
            try:
                # Tokenize with start and end tokens
                tokens = self.tokenizer.tokenize(smi, with_begin_and_end=True)
                
                # Filter by length
                if len(tokens) > max_length:
                    n_filtered_length += 1
                    continue
                
                # Check all tokens are in vocabulary
                if not all(t in self.vocabulary for t in tokens):
                    n_filtered_vocab += 1
                    continue
                
                # Encode tokens to indices
                encoded = self.vocabulary.encode(tokens)
                
                self.smiles_list.append(smi)
                self.encoded_sequences.append(torch.tensor(encoded, dtype=torch.long))
                
            except Exception as e:
                continue
        
        print(f"SMILESDataset: Loaded {len(self.smiles_list)} valid SMILES")
        print(f"  - Filtered {n_filtered_length} (too long, >{max_length})")
        print(f"  - Filtered {n_filtered_vocab} (unknown tokens)")
    
    def __len__(self) -> int:
        return len(self.smiles_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            input_seq: Tensor of shape (seq_len-1,) - sequence without last token
            target_seq: Tensor of shape (seq_len-1,) - sequence without first token
        """
        encoded = self.encoded_sequences[idx]
        
        # Input: everything except last token [^, t1, t2, ..., tN]
        input_seq = encoded[:-1]
        
        # Target: everything except first token [t1, t2, ..., tN, $]
        target_seq = encoded[1:]
        
        return input_seq, target_seq
    
    def get_smiles(self, idx: int) -> str:
        """Get the original SMILES string at index."""
        return self.smiles_list[idx]
    
    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate function for DataLoader that pads sequences.
        
        CRITICAL FIX: 
        - Inputs are padded with 0 (EOS token, fine for inputs)
        - Targets are padded with -100 (PyTorch's ignore_index default)
        
        This ensures EOS (index 0) is LEARNED in targets, while padding is ignored!
        Previously, both EOS and padding were 0, so EOS was never learned.
        
        Args:
            batch: List of (input_seq, target_seq) tuples
            
        Returns:
            inputs: Padded input sequences (batch_size, max_seq_len) - padded with 0
            targets: Padded target sequences (batch_size, max_seq_len) - padded with -100
        """
        inputs, targets = zip(*batch)
        
        # Find max length in batch
        max_len = max(seq.size(0) for seq in inputs)
        
        batch_size = len(inputs)
        
        # Inputs: pad with 0 (EOS token - fine for input side)
        padded_inputs = torch.zeros(batch_size, max_len, dtype=torch.long)
        
        # CRITICAL: Targets must be padded with -100 (not 0!)
        # -100 is PyTorch's default ignore_index for CrossEntropyLoss
        # This way, EOS (index 0) is LEARNED, but padding is IGNORED
        padded_targets = torch.full((batch_size, max_len), fill_value=-100, dtype=torch.long)
        
        for i, (inp, tgt) in enumerate(zip(inputs, targets)):
            seq_len = inp.size(0)
            padded_inputs[i, :seq_len] = inp
            padded_targets[i, :seq_len] = tgt
        
        return padded_inputs, padded_targets


def load_smiles_from_file(filepath: str) -> List[str]:
    """
    Load SMILES strings from a text file.
    
    Handles various formats:
    - One SMILES per line
    - SMILES with ID/name (space or tab separated, SMILES first)
    - Lines starting with # are skipped
    
    Args:
        filepath: Path to the SMILES file
        
    Returns:
        List of SMILES strings
    """
    smiles = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Handle different formats (SMILES may be followed by ID)
            parts = line.split()
            if parts:
                smi = parts[0]
                # Basic validation: must contain at least one letter
                if any(c.isalpha() for c in smi):
                    smiles.append(smi)
    
    return smiles


def create_dataloader(
    smiles_file: str,
    vocabulary: Vocabulary,
    tokenizer: SMILESTokenizer,
    batch_size: int = 64,
    max_length: int = 100,
    shuffle: bool = True,
    num_workers: int = 0,
    val_split: float = 0.1,
    seed: int = 42,
) -> Tuple:
    """
    Create train and validation DataLoaders from a SMILES file.
    
    Args:
        smiles_file: Path to SMILES file
        vocabulary: Vocabulary object
        tokenizer: SMILESTokenizer object
        batch_size: Batch size for DataLoader
        max_length: Maximum sequence length
        shuffle: Whether to shuffle training data
        num_workers: Number of DataLoader workers
        val_split: Fraction of data to use for validation
        seed: Random seed for splitting
        
    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    from torch.utils.data import DataLoader, random_split
    
    # Load SMILES
    smiles_list = load_smiles_from_file(smiles_file)
    print(f"Loaded {len(smiles_list)} SMILES from {smiles_file}")
    
    # Create dataset
    dataset = SMILESDataset(
        smiles_list,
        vocabulary,
        tokenizer,
        max_length=max_length,
    )
    
    # Split into train/val
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=generator)
    
    print(f"Train: {n_train} samples, Val: {n_val} samples")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=SMILESDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=SMILESDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    return train_loader, val_loader, train_dataset, val_dataset


if __name__ == "__main__":
    # Test the dataset
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SMILESDataset")
    parser.add_argument("--data", type=str, required=True, help="Path to SMILES file")
    parser.add_argument("--prior", type=str, required=True, help="Path to prior model (for vocabulary)")
    args = parser.parse_args()
    
    # Load vocabulary from prior
    prior_dict = torch.load(args.prior, map_location='cpu')
    vocabulary = prior_dict['vocabulary']
    tokenizer = prior_dict.get('tokenizer', SMILESTokenizer())
    
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Start token '^' = {vocabulary['^']}")
    print(f"End token '$' = {vocabulary['$']}")
    
    # Create dataset
    smiles_list = load_smiles_from_file(args.data)
    dataset = SMILESDataset(smiles_list, vocabulary, tokenizer)
    
    # Test a sample
    if len(dataset) > 0:
        inp, tgt = dataset[0]
        print(f"\nSample 0:")
        print(f"  SMILES: {dataset.get_smiles(0)}")
        print(f"  Input shape: {inp.shape}")
        print(f"  Target shape: {tgt.shape}")
        print(f"  Input tokens: {vocabulary.decode(inp.numpy())}")
        print(f"  Target tokens: {vocabulary.decode(tgt.numpy())}")
