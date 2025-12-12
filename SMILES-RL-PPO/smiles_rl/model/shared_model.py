"""
Implementation of wrapper for shared Transformer model
"""
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as tnn


from .vocabulary import Vocabulary, create_vocabulary_from_hf_tokenizer
from .smiles_tokenizer import SMILESTokenizer, ChemGPTTokenizerAdapter


from .transformer import ChemGPTTransformerShared


class SharedModel:
    """
    Implements a wrapper for shared Transformer model using SMILES.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        tokenizer,
        network_params=None,
        max_sequence_length: int = 256,
        no_cuda: bool = False,
    ):
        """
        Implements a shared Transformer model.
        :param vocabulary: Vocabulary to use.
        :param tokenizer: Tokenizer to use.
        :param network_params: Dictionary with all parameters required to correctly initialize the network.
        :param max_sequence_length: The max size of SMILES sequence that can be generated.
        """
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self._network_params = dict(network_params) if isinstance(network_params, dict) else {}

        network_params = dict(network_params) if isinstance(network_params, dict) else {}
        self.backend = network_params.pop("backend", "transformer").lower()

        model_name_or_path = network_params.pop(
            "model_name_or_path", "ncfrey/ChemGPT-19M"
        )
        fine_tune = network_params.pop("fine_tune", True)
        layer_normalization = network_params.pop("layer_normalization", False)
        use_cache = network_params.pop("use_cache", True)

        if tokenizer is None or isinstance(tokenizer, SMILESTokenizer):
            self.tokenizer = ChemGPTTokenizerAdapter(model_name_or_path)
        if not isinstance(self.vocabulary, Vocabulary):
            self.vocabulary = create_vocabulary_from_hf_tokenizer(
                self.tokenizer.tokenizer
            )

        pad_token_id = None
        try:
            pad_token_id = self.vocabulary["$"]
        except KeyError:
            pad_token_id = getattr(self.tokenizer.tokenizer, "pad_token_id", None)

        self.network = ChemGPTTransformerShared(
            model_name_or_path=model_name_or_path,
            pad_token_id=pad_token_id,
            fine_tune=fine_tune,
            layer_normalization=layer_normalization,
            use_cache=use_cache,
        )

        if torch.cuda.is_available() and not no_cuda:
            self.network.cuda()

        self._nll_loss = tnn.NLLLoss(reduction="none")

    def reset_policy_output_layer(
        self,
    ):
        if hasattr(self.network, "_linear"):
            self.network._linear = tnn.Linear(
                self.network._layer_size, len(self.vocabulary)
            )
        elif hasattr(self.network, "_policy_head"):
            hidden_size = self.network._model.config.hidden_size
            self.network._policy_head = tnn.Linear(hidden_size, len(self.vocabulary))

    def reset_critic_output_layer(
        self,
    ):
        if hasattr(self.network, "_critic"):
            self.network._critic = tnn.Linear(
                self.network._layer_size, len(self.vocabulary)
            )
        elif hasattr(self.network, "_critic_head"):
            hidden_size = self.network._model.config.hidden_size
            self.network._critic_head = tnn.Linear(
                hidden_size, len(self.vocabulary)
            )

    def set_mode(self, mode: str):
        if mode == "training":
            self.network.train()
        elif mode == "inference":
            self.network.eval()
        else:
            raise ValueError(f"Invalid model mode '{mode}")

    @classmethod
    def load_from_file(cls, file_path: str, sampling_mode: bool = False):
        """
        Loads a model from a single file
        :param file_path: input file path
        :return: new instance of the RNN or an exception if it was not possible to load it.
        """
        if torch.cuda.is_available():
            save_dict = torch.load(file_path, weights_only=False)
        else:
            save_dict = torch.load(file_path, map_location=lambda storage, loc: storage, weights_only=False)

        network_params = save_dict.get("network_params", {})
        model = SharedModel(
            vocabulary=save_dict["vocabulary"],
            tokenizer=save_dict.get("tokenizer", SMILESTokenizer()),
            network_params=network_params,
            max_sequence_length=save_dict["max_sequence_length"],
        )

        # Pretrained model only has one head: the policy output
        # This step adds the random initialized parameters for the critic model,
        # so that we can load the pre-trained paramaters into the model
        pretrained_dict = save_dict["network"]
        model_dict = model.network.state_dict()
        model_dict.update(pretrained_dict)
        model.network.load_state_dict(model_dict)

        if sampling_mode:
            model.set_mode("inference")
        else:
            model.set_mode("training")

        return model

    def save(self, file: str):
        """
        Saves the model into a file
        :param file: it's actually a path
        """
        save_dict = {
            "vocabulary": self.vocabulary,
            "tokenizer": self.tokenizer,
            "max_sequence_length": self.max_sequence_length,
            "network": self.network.state_dict(),
            "network_params": self.network.get_params(),
        }
        torch.save(save_dict, file)

    def smiles_to_sequences(self, smiles: List[str]) -> torch.Tensor:
        end_token = self.vocabulary["$"]
        tokens = [
            self.tokenizer.tokenize(smile, with_begin_and_end=True) for smile in smiles
        ]
        encoded = [self.vocabulary.encode(token) for token in tokens]
        sequences = [torch.tensor(encode, dtype=torch.long) for encode in encoded]

        def collate_fn(encoded_seqs: List[torch.Tensor]) -> torch.Tensor:
            """Function to take a list of encoded sequences and turn them into a batch"""
            max_length = max([seq.size(0) for seq in encoded_seqs])
            collated_arr = torch.full(
                (len(encoded_seqs), max_length), end_token, dtype=torch.long
            )
            for i, seq in enumerate(encoded_seqs):
                collated_arr[i, : seq.size(0)] = seq
            return collated_arr

        padded_sequences = collate_fn(sequences)

        return padded_sequences.to("cuda")

    def likelihood_smiles(self, smiles: List[str]) -> torch.Tensor:
        end_token = self.vocabulary["$"]
        tokens = [self.tokenizer.tokenize(smile) for smile in smiles]
        encoded = [self.vocabulary.encode(token) for token in tokens]
        sequences = [torch.tensor(encode, dtype=torch.long) for encode in encoded]

        def collate_fn(encoded_seqs):
            """Function to take a list of encoded sequences and turn them into a batch"""
            max_length = max([seq.size(0) for seq in encoded_seqs])
            collated_arr = torch.full(
                (len(encoded_seqs), max_length), end_token, dtype=torch.long
            )
            for i, seq in enumerate(encoded_seqs):
                collated_arr[i, : seq.size(0)] = seq
            return collated_arr

        padded_sequences = collate_fn(sequences)
        return self.likelihood(padded_sequences)

    def likelihood(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Retrieves the (log) likelihood of a given sequence. Used in training.

        :param sequences: (batch_size, sequence_length) A batch of sequences
        :return:  (batch_size) Log likelihood for each example.
        """
        logits, _, _ = self.network(sequences[:, :-1])  # all steps done at once
        log_probs = logits.log_softmax(dim=2)
        return self._nll_loss(log_probs.transpose(1, 2), sequences[:, 1:]).sum(dim=1)

    def sample_smiles(self, num=128, batch_size=128) -> Tuple[List, np.ndarray]:
        """
        Samples n SMILES from the model.
        :param num: Number of SMILES to sample.
        :param batch_size: Number of sequences to sample at the same time.
        :return:
            :smiles: (n) A list with SMILES.
            :likelihoods: (n) A list of likelihoods.
        """
        batch_sizes = [batch_size for _ in range(num // batch_size)] + [
            num % batch_size
        ]
        smiles_sampled = []
        batch_probs_sampled = []

        for size in batch_sizes:
            if not size:
                break
            seqs, batch_probs = self._sample(batch_size=size)
            smiles = [
                self.tokenizer.untokenize(self.vocabulary.decode(seq))
                for seq in seqs.cpu().numpy()
            ]

            smiles_sampled.extend(smiles)
            batch_probs_sampled.append(batch_probs.data.cpu().numpy())

            del seqs, batch_probs
        return smiles_sampled, np.concatenate(batch_probs_sampled, axis=0)

    def sample_sequences_and_smiles(
        self, batch_size=128
    ) -> Tuple[torch.Tensor, List, torch.Tensor]:
        seqs, batch_probs = self._sample(batch_size=batch_size)
        smiles = [
            self.tokenizer.untokenize(self.vocabulary.decode(seq))
            for seq in seqs.cpu().numpy()
        ]
        return seqs, smiles, batch_probs

    # @torch.no_grad()
    def _sample(self, batch_size=128) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.network.parameters()).device
        start_token = torch.zeros(batch_size, dtype=torch.long, device=device)
        start_token[:] = self.vocabulary["^"]
        end_token = torch.tensor(self.vocabulary["$"], device=device)
        input_vector = start_token
        sequences = [
            self.vocabulary["^"]
            * torch.ones([batch_size, 1], dtype=torch.long, device=device)
        ]
        batch_probs = []
        # NOTE: The first token never gets added in the loop so the sequences are initialized with a start token
        hidden_state = None
        for _ in range(self.max_sequence_length - 1):
            logits, _, hidden_state = self.network(
                input_vector.unsqueeze(1), hidden_state
            )
            logits = logits.squeeze(1)  # .detach() # NOTE: Added detach
            probabilities = logits.softmax(dim=1)

            input_vector = torch.multinomial(probabilities, 1).view(-1)
            batch_probs.append(probabilities.unsqueeze(1))
            sequences.append(input_vector.view(-1, 1))

            if torch.all(input_vector == end_token):
                break

        sequences = torch.cat(sequences, 1)
        probs = torch.cat(batch_probs, 1)

        assert probs.size() == (
            batch_size,
            sequences.size(1) - 1,
            len(self.vocabulary),
        ), f"log probs {probs.size()}, correct {(sequences.size(0),sequences.size(1),len(self.vocabulary))}"

        # TODO: previously these were detached at return
        return sequences, probs

    def log_probabilities(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Retrieves the log probabilities of all actions given sequence.

        :param sequences: (batch_size, sequence_length) A batch of sequences
        :return:  (batch_size, sequence_length-1, num_actions) Log probabilities for action in sequence.
        """
        # Excluding last token for consistency with likelihoods
        seqs = sequences[:, :-1]

        logits, _, _ = self.network(seqs, None)  # all steps done at once

        log_probs = logits.log_softmax(dim=-1)

        assert log_probs.size() == (
            sequences.size(0),
            sequences.size(1) - 1,
            len(self.vocabulary),
        ), f"log probs {log_probs.size()}, correct {(sequences.size(0),sequences.size(1),len(self.vocabulary))}"

        return log_probs

    def log_and_probabilities(
        self, sequences: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the log probabilities of all actions given sequence.

        :param sequences: (batch_size, sequence_length) A batch of sequences
        :return:  (batch_size, sequence_length-1, num_actions) Log probabilities for action in sequence.
        """

        # Excluding last token for consistency with likelihoods
        seqs = sequences[:, :-1]

        logits, _, _ = self.network(seqs, None)  # all steps done at once

        log_probs = logits.log_softmax(dim=-1)
        probs = logits.softmax(dim=-1)

        assert log_probs.size() == (
            sequences.size(0),
            sequences.size(1) - 1,
            len(self.vocabulary),
        ), f"log probs {log_probs.size()}, correct {(sequences.size(0),sequences.size(1),len(self.vocabulary))}"

        return log_probs, probs

    def probabilities(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Retrieves the probabilities of all actions given sequence.

        :param sequences: (batch_size, sequence_length) A batch of sequences
        :return:  (batch_size, sequence_length-1, num_actions) Probabilities for action in sequence.
        """

        # Excluding last token for consistency with likelihoods
        seqs = sequences[:, :-1]

        logits, _, _ = self.network(seqs, None)  # all steps done at once

        probs = logits.softmax(dim=-1)

        assert probs.size() == (
            sequences.size(0),
            sequences.size(1) - 1,
            len(self.vocabulary),
        ), f"log probs {probs.size()}, correct {(sequences.size(0),sequences.size(1),len(self.vocabulary))}"

        return probs

    def log_probabilities_action(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Retrieves the log probabilities of action taken a given sequence.

        :param sequences: (batch_size, sequence_length) A batch of sequences
        :return:  (batch_size, sequence_length-1) Log probabilities for action in sequence.
        """
        # Remove last action of sequences (stop token)
        seqs = sequences[:, :-1]

        logits, _, _ = self.network(seqs, None)  # all steps done at once

        log_probs = logits.log_softmax(dim=-1)

        log_probs = torch.gather(log_probs, -1, sequences[:, 1:].unsqueeze(-1)).squeeze(
            -1
        )

        assert (
            log_probs.size() == seqs.size()
        ), f"log probs {log_probs.size()}, seqs {seqs.size()}"

        return log_probs

    def q_values(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Retrieves the state action values for each action in given sequence.

        :param sequences: (batch_size, sequence_length) A batch of sequences
        :return:  (batch_size, sequence_length-1, n_actions) q-values (logits) for each possible action in sequence.
        """

        # Excluding last token for consistency with likelihoods
        seqs = sequences[:, :-1]

        _, q_values, _ = self.network(seqs, None)  # all steps done at once

        assert q_values.size() == (
            seqs.size(0),
            seqs.size(1),
            len(self.vocabulary),
        ), f"q_values has incorrect shape {q_values.size()}, should be {(seqs.size(0), seqs.size(1), len(self.vocabulary))}"

        return q_values

    def get_network_parameters(self):
        return self.network.parameters()

    def save_to_file(self, path: str):
        self.save(path)

    def sample(self, batch_size: int):
        return self.sample_sequences_and_smiles(batch_size)

    def get_vocabulary(self):
        return self.vocabulary

    def load_state_dict(self, state_dict: dict):
        self.network.load_state_dict(state_dict)

    def state_dict(
        self,
    ):
        return self.network.state_dict()
