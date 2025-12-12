from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as tnn
import os
from copy import deepcopy

from .vocabulary import Vocabulary
from .smiles_tokenizer import SMILESTokenizer
from .transformer import Transformer


class ActorModelTransformer:
    """
    Implements a Transformer model using SMILES for Actor.
    Used as student model in knowledge transfer from RNN (teacher).
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        tokenizer,
        network_params=None,
        max_sequence_length=256,
        no_cuda=False,
    ):
        """
        Implements a Transformer actor model.
        :param vocabulary: Vocabulary to use.
        :param tokenizer: Tokenizer to use.
        :param network_params: Dictionary with all parameters required to correctly initialize the Transformer class.
        :param max_sequence_length: The max size of SMILES sequence that can be generated.
        """
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

        if not isinstance(network_params, dict):
            network_params = {}

        # Normalize/standardize network params (accept legacy keys)
        network_params = self._standardize_network_params(network_params)

        # Default transformer parameters if not specified
        voc_size = len(self.vocabulary)
        layer_size = network_params.get("layer_size", 256)
        n_layers = network_params.get("n_layers", 6)
        n_heads = network_params.get("n_heads", 8)
        dropout = network_params.get("dropout", 0.0)

        self.transformer = Transformer(
            voc_size=voc_size,
            max_seq_len=max_sequence_length,
            layer_size=layer_size,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        
        if torch.cuda.is_available() and not no_cuda:
            self.transformer.cuda()

        self._nll_loss = tnn.NLLLoss(reduction="none")

    @classmethod
    def load_from_file(
        cls,
        pre_training_file_path: str,
        transfer_weight_path: str = None,
        sampling_mode: bool = False,
        network_params_override: Optional[dict] = None,
        freeze_embeddings: bool = False,
        max_sequence_length: Optional[int] = None,
    ):
        """
        Loads a transformer model from a pre-trained RNN file.
        Uses the vocabulary and tokenizer from the RNN model.
        Optionally loads transformer weights if transfer_weight_path is provided.
        
        :param pre_training_file_path: Path to pre-trained RNN model (for vocabulary/tokenizer)
        :param transfer_weight_path: Optional path to pre-trained transformer weights
        :param sampling_mode: Whether to set model to inference mode
        :return: new instance of ActorModelTransformer
        """
        # Load the RNN model to get vocabulary and tokenizer
        if torch.cuda.is_available():
            rnn_save_dict = torch.load(pre_training_file_path)
        else:
            rnn_save_dict = torch.load(pre_training_file_path, map_location=lambda storage, loc: storage)

        # Check if we have a transformer checkpoint to load
        # If so, get network_params from the CHECKPOINT, not the RNN prior
        # Allow explicit override of architecture (e.g., paper spec)
        network_params = network_params_override or rnn_save_dict.get("network_params", {})
        max_seq_len = max_sequence_length or rnn_save_dict["max_sequence_length"]
        
        if transfer_weight_path and os.path.exists(transfer_weight_path):
            if torch.cuda.is_available():
                transformer_state = torch.load(transfer_weight_path)
            else:
                transformer_state = torch.load(transfer_weight_path, map_location=lambda storage, loc: storage)
            
            # CRITICAL FIX: Use network_params from the checkpoint if available
            # This ensures we create the model with the SAME architecture as the checkpoint
            if isinstance(transformer_state, dict):
                if "network_params" in transformer_state and network_params_override is None:
                    network_params = transformer_state["network_params"]
                    print(f"  [load_from_file] Using network_params from checkpoint: {network_params}")
                if "max_sequence_length" in transformer_state and max_sequence_length is None:
                    max_seq_len = transformer_state["max_sequence_length"]
        else:
            transformer_state = None

        # Create transformer model with correct architecture
        model = ActorModelTransformer(
            vocabulary=rnn_save_dict["vocabulary"],
            tokenizer=rnn_save_dict.get("tokenizer", SMILESTokenizer()),
            network_params=network_params,
            max_sequence_length=max_seq_len,
        )

        # Load transformer weights if provided
        if transformer_state is not None:
            # Handle both raw transformer state_dict and full save_dict with metadata
            if isinstance(transformer_state, dict) and "network" in transformer_state:
                transformer_state = transformer_state["network"]

            model.transformer.load_state_dict(transformer_state, strict=True)

        if sampling_mode:
            model.transformer.eval()
        else:
            model.transformer.train()

        if freeze_embeddings and hasattr(model.transformer, "freeze_layers"):
            model.transformer.freeze_layers(["embed"])

        return model

    def set_mode(self, mode: str):
        if mode == "training":
            self.transformer.train()
        elif mode == "inference":
            self.transformer.eval()
        else:
            raise ValueError(f"Invalid model mode '{mode}")

    def q_values(self, sequences: torch.Tensor):
        """
        Retrieves the state action values for each action in given sequence.

        :param sequences: (batch_size, sequence_length) A batch of sequences
        :return: (batch_size, sequence_length-1, n_actions) q-values (logits) for each possible action in sequence.
        """
        # Excluding last token for consistency
        seqs = sequences[:, :-1]
        
        batch_size, seq_len = seqs.size()
        device = seqs.device

        # Create causal mask for transformer.
        # attention() expects a mask shaped (batch, seq_len, seq_len); it will unsqueeze(1)
        # to (batch, 1, seq_len, seq_len), which then broadcasts across heads.
        causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        mask = causal.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, seq_len, seq_len)

        q_values = self.transformer(seqs, trg_mask=mask)

        assert q_values.size() == (
            seqs.size(0),
            seqs.size(1),
            len(self.vocabulary),
        ), f"q_values has incorrect shape {q_values.size()}, should be {(seqs.size(0), seqs.size(1), len(self.vocabulary))}"

        return q_values

    def state_dict(self):
        """Returns the state dict of the transformer model."""
        return self.transformer.state_dict()

    def save(self, file: str):
        """Persist transformer weights and metadata."""
        save_dict = {
            "vocabulary": self.vocabulary,
            "tokenizer": self.tokenizer,
            "max_sequence_length": self.max_sequence_length,
            "network": self.transformer.state_dict(),
            "network_params": self.transformer.get_params(),
        }
        torch.save(save_dict, file)

    def save_to_file(self, path: str):
        """Compatibility with logger expectations."""
        self.save(path)

    # -------- RNN-API parity helpers --------
    def reset_output_layer(self):
        # Re-initialize output projection
        tnn.init.xavier_uniform_(self.transformer.out.weight)
        if self.transformer.out.bias is not None:
            tnn.init.zeros_(self.transformer.out.bias)

    def likelihood_smiles(self, smiles: List[str]) -> torch.Tensor:
        seqs = self.smiles_to_sequences(smiles)
        return self.likelihood(seqs)

    def sample_smiles(self, num=128, batch_size=128) -> Tuple[List[str], np.ndarray]:
        batch_sizes = [batch_size for _ in range(num // batch_size)] + [num % batch_size]
        smiles_sampled = []
        likelihoods_sampled = []

        for size in batch_sizes:
            if not size:
                break
            seqs, smiles, likelihoods = self.sample(batch_size=size)
            smiles_sampled.extend(smiles)
            likelihoods_sampled.append(likelihoods.data.cpu().numpy())
            del seqs, likelihoods
        return smiles_sampled, np.concatenate(likelihoods_sampled)

    def sample_sequences_and_smiles(
        self, batch_size=128
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        return self.sample(batch_size)

    def log_probabilities(self, sequences: torch.Tensor):
        seqs = sequences[:, :-1]
        logits = self.transformer(seqs)
        log_probs = logits.log_softmax(dim=-1)
        return log_probs

    def log_and_probabilities(self, sequences: torch.Tensor):
        logits = self.transformer(sequences[:, :-1])
        log_probs = logits.log_softmax(dim=2)
        probs = log_probs.exp()
        return log_probs, probs

    def log_and_probabilities_action(self, sequences: torch.Tensor):
        logits = self.transformer(sequences[:, :-1])
        log_probs = logits.log_softmax(dim=2)
        probs = logits.softmax(dim=2)
        log_probs_action = torch.gather(log_probs, -1, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
        probs_action = torch.gather(probs, -1, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
        return log_probs_action, probs_action

    def probabilities(self, sequences: torch.Tensor):
        logits = self.transformer(sequences[:, :-1])
        probs = logits.softmax(dim=2)
        return probs

    def log_probabilities_action(self, sequences: torch.Tensor):
        logits = self.transformer(sequences[:, :-1])
        log_probs = logits.log_softmax(dim=2)
        log_probs = torch.gather(log_probs, -1, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
        return log_probs

    def load_state_dict(self, state_dict: dict):
        """Loads state dict into the transformer model."""
        self.transformer.load_state_dict(state_dict)

    def get_vocabulary(self):
        return self.vocabulary

    def get_network_parameters(self):
        return self.transformer.parameters()

    # --------- Additional helpers to mirror the RNN actor API ----------
    def smiles_to_sequences(self, smiles: List[str]) -> torch.Tensor:
        """Tokenize and pad SMILES strings to a tensor batch on the current device."""
        tokens = [
            self.tokenizer.tokenize(smile, with_begin_and_end=True) for smile in smiles
        ]
        encoded = [self.vocabulary.encode(token) for token in tokens]
        sequences = [torch.tensor(encode, dtype=torch.long) for encode in encoded]

        def collate_fn(encoded_seqs: List[torch.Tensor]) -> torch.Tensor:
            max_length = max([seq.size(0) for seq in encoded_seqs])
            collated_arr = torch.zeros(
                len(encoded_seqs), max_length, dtype=torch.long
            )  # padded with zeroes
            for i, seq in enumerate(encoded_seqs):
                collated_arr[i, : seq.size(0)] = seq
            return collated_arr

        padded_sequences = collate_fn(sequences)
        return padded_sequences.to(next(self.transformer.parameters()).device)

    def log_and_probabilities(self, sequences: torch.Tensor):
        """Return log-probs and probs over vocabulary for each position."""
        logits = self.transformer(sequences[:, :-1])
        log_probs = logits.log_softmax(dim=2)
        probs = log_probs.exp()
        return log_probs, probs

    def likelihood(self, sequences: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood per sequence."""
        logits = self.transformer(sequences[:, :-1])
        log_probs = logits.log_softmax(dim=2)
        return self._nll_loss(log_probs.transpose(1, 2), sequences[:, 1:]).sum(dim=1)

    def sample(self, batch_size: int, min_length: int = 5):
        """
        Sample sequences and smiles, returning (sequences, smiles, log_probs).
        min_length: prevent EOS before this many tokens (including start).
        Additionally, do not allow EOS as the very first generated token.
        """
        self.set_mode("inference")
        device = next(self.transformer.parameters()).device
        start_token = torch.full((batch_size,), self.vocabulary["^"], dtype=torch.long, device=device)

        input_vector = start_token
        sequences = [start_token.view(batch_size, 1)]
        batch_log_probs = []

        for step in range(self.max_sequence_length - 1):
            current_seq = torch.cat(sequences, dim=1)
            logits = self.transformer(current_seq)
            logits = logits[:, -1, :]  # last token
            probabilities = logits.softmax(dim=1)
            log_probs = logits.log_softmax(dim=1)

            # Disallow EOS on first step; mask EOS until min_length reached
            if step == 0:
                eos_idx = 0
                probs_clone = probabilities.clone()
                probs_clone[:, eos_idx] = 0.0
                row_sums = probs_clone.sum(dim=1, keepdim=True)
                mask_nonzero = row_sums.squeeze(1) > 0
                if mask_nonzero.any():
                    probabilities = torch.where(
                        mask_nonzero.unsqueeze(1),
                        probs_clone / row_sums.clamp_min(1e-9),
                        probabilities,
                    )
            elif step + 1 < min_length:
                eos_idx = 0
                probs_clone = probabilities.clone()
                probs_clone[:, eos_idx] = 0.0
                row_sums = probs_clone.sum(dim=1, keepdim=True)
                mask_nonzero = row_sums.squeeze(1) > 0
                if mask_nonzero.any():
                    probabilities = torch.where(
                        mask_nonzero.unsqueeze(1),
                        probs_clone / row_sums.clamp_min(1e-9),
                        probabilities,
                    )

            input_vector = torch.multinomial(probabilities, 1).view(-1)

            batch_log_probs.append(log_probs.gather(1, input_vector.view(-1, 1)))
            sequences.append(input_vector.view(-1, 1))

            if input_vector.sum() == 0 and step + 1 >= min_length:  # all EOS
                break

        sequences = torch.cat(sequences, 1)
        batch_log_probs = torch.cat(batch_log_probs, 1)

        smiles = [
            self.tokenizer.untokenize(self.vocabulary.decode(seq))
            for seq in sequences.cpu().numpy()
        ]

        self.set_mode("training")
        return sequences, smiles, batch_log_probs

    def freeze_layers(self, layers_to_freeze):
        self.transformer.freeze_layers(layers_to_freeze)

    @staticmethod
    def _standardize_network_params(network_params: Optional[dict]) -> dict:
        """Ensure consistent keys for transformer architecture."""
        network_params = network_params or {}
        layer_size = network_params.get("layer_size", network_params.get("d_model", 256))
        n_layers = network_params.get("n_layers", network_params.get("num_layers", 6))
        n_heads = network_params.get("n_heads", network_params.get("num_heads", 8))
        dropout = network_params.get("dropout", 0.0)
        return {
            "layer_size": layer_size,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "dropout": dropout,
        }

