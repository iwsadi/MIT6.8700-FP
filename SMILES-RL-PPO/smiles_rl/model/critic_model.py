import numpy as np
import torch
import torch.nn as tnn

from .vocabulary import Vocabulary, create_vocabulary_from_hf_tokenizer
from .smiles_tokenizer import SMILESTokenizer, ChemGPTTokenizerAdapter


from .transformer import ChemGPTTransformer


class CriticModel:
    """
    Implements a Transformer model using SMILES for Critic (value function).
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
        Implements a Transformer for the critic.
        :param vocabulary: Vocabulary to use.
        :param tokenizer: Tokenizer to use.
        :param network_params: Dictionary with all parameters required to correctly initialize the network.
        :param max_sequence_length: The max size of SMILES sequence that can be generated.
        """
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

        network_params = dict(network_params) if isinstance(network_params, dict) else {}
        self.backend = network_params.pop("backend", "transformer").lower()

        model_name_or_path = network_params.pop("model_name_or_path", "ncfrey/ChemGPT-19M")
        fine_tune = network_params.pop("fine_tune", True)
        layer_normalization = network_params.pop("layer_normalization", False)
        use_cache = network_params.pop("use_cache", True)

        # Use ChemGPT tokenizer/vocab if not already set
        if tokenizer is None or isinstance(tokenizer, SMILESTokenizer):
            self.tokenizer = ChemGPTTokenizerAdapter(model_name_or_path)
        if not isinstance(self.vocabulary, Vocabulary) or len(self.vocabulary) == 0:
            self.vocabulary = create_vocabulary_from_hf_tokenizer(self.tokenizer.tokenizer)

        pad_token_id = None
        try:
            pad_token_id = self.vocabulary["$"]
        except KeyError:
            pad_token_id = getattr(self.tokenizer.tokenizer, "pad_token_id", None)

        # Critic outputs single value per position
        self.network = ChemGPTTransformer(
            model_name_or_path=model_name_or_path,
            pad_token_id=pad_token_id,
            fine_tune=fine_tune,
            layer_normalization=layer_normalization,
            use_cache=use_cache,
            output_size=1,
        )

        if torch.cuda.is_available() and not no_cuda:
            self.network.cuda()

        self._nll_loss = tnn.NLLLoss(reduction="none")

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
        :return: new instance of the RNN/Transformer or an exception if it was not possible to load it.
        """
        if torch.cuda.is_available():
            save_dict = torch.load(file_path, weights_only=False)
        else:
            save_dict = torch.load(file_path, map_location=lambda storage, loc: storage, weights_only=False)

        network_params = save_dict.get("network_params", {})
        
        # For critic, we need to override output_size to 1 for value function
        # but keep the backend info from the checkpoint
        critic_params = dict(network_params)
        critic_params["output_size"] = 1
        
        model = CriticModel(
            vocabulary=save_dict["vocabulary"],
            tokenizer=save_dict.get("tokenizer", SMILESTokenizer()),
            network_params=critic_params,
            max_sequence_length=save_dict["max_sequence_length"],
        )

        # Load saved weights where shapes match (e.g., backbone and value head if compatible)
        saved_state = save_dict.get("network")
        if saved_state is not None:
            model_state = model.network.state_dict()
            filtered_state = {
                k: v for k, v in saved_state.items()
                if k in model_state and model_state[k].shape == v.shape
            }
            if filtered_state:
                model_state.update(filtered_state)
                model.network.load_state_dict(model_state)

        if torch.cuda.is_available():
            model.network.cuda()

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

    def values(self, sequences: torch.Tensor):
        """
        Retrieves the action-values of a given sequence.

        :param sequences: (batch_size, sequence_length) A batch of sequences
        :return:  (batch_size, sequence_length-1) action-value for each batch.
        """

        # All steps done at once
        # Excluding last token for consistency with likelihoods
        value, _ = self.network(sequences[:, :-1])

        return value.squeeze(-1)

    def get_network_parameters(self):
        return self.network.parameters()

    def get_parameter_groups(self, lr_backbone: float, lr_head: float):
        """
        Returns parameter groups with different learning rates for backbone and head.
        
        Args:
            lr_backbone: Learning rate for the pretrained backbone
            lr_head: Learning rate for the output head
            
        Returns:
            List of parameter group dicts suitable for torch.optim.Adam
        """
        return self.network.get_parameter_groups(lr_backbone, lr_head)

    def freeze_backbone(self):
        """Freeze backbone parameters, only allow head to be trained."""
        if hasattr(self.network, 'freeze_backbone'):
            self.network.freeze_backbone()

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters to allow fine-tuning."""
        if hasattr(self.network, 'unfreeze_backbone'):
            self.network.unfreeze_backbone()

    def save_to_file(self, path: str):
        self.save(path)

    def get_vocabulary(self):
        return self.vocabulary

    def load_state_dict(self, state_dict: dict):
        self.network.load_state_dict(state_dict)

    def state_dict(
        self,
    ):
        return self.network.state_dict()
