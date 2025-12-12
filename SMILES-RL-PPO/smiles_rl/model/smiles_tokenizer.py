import re
from typing import List

try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover - optional dependency
    AutoTokenizer = None
    _transformers_import_error = exc
else:
    _transformers_import_error = None

class SMILESTokenizer:
    """Deals with the tokenization and untokenization of SMILES."""

    REGEXPS = {
        "brackets": re.compile(r"(\[[^\]]*\])"),
        "2_ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl)"),
    }
    REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]

    def tokenize(self, data, with_begin_and_end=True):
        """Tokenizes a SMILES string."""

        def split_by(data, regexps):
            if not regexps:
                return list(data)
            regexp = self.REGEXPS[regexps[0]]
            splitted = regexp.split(data)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens

        tokens = split_by(data, self.REGEXP_ORDER)
        if with_begin_and_end:
            tokens = ["^"] + tokens + ["$"]
        return tokens

    def untokenize(self, tokens):
        """Untokenizes a SMILES string."""
        smi = ""
        for token in tokens:
            if token == "$":
                break
            if token != "^":
                smi += token
        return smi


class ChemGPTTokenizerAdapter:
    """
    Wraps a Hugging Face ChemGPT tokenizer to match the SMILES tokenizer API
    expected by the rest of the codebase.
    """

    def __init__(self, model_name_or_path: str):
        if AutoTokenizer is None:
            raise ImportError(
                "transformers is required for ChemGPT tokenizer"
            ) from _transformers_import_error

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Ensure special tokens exist for BOS/EOS/PAD.
        if self.tokenizer.bos_token is None:
            self.tokenizer.add_special_tokens({"bos_token": "^"})
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "$"})
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "$"})

        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.pad_token = self.tokenizer.pad_token

    def tokenize(self, data: str, with_begin_and_end: bool = True) -> List[str]:
        tokens = self.tokenizer.tokenize(data)
        if with_begin_and_end:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        return tokens

    def untokenize(self, tokens: List[str]) -> str:
        # Tokens that indicate start of sequence (skip them)
        bos_tokens = {self.bos_token, self.tokenizer.bos_token, "^", "<bos>"}
        # Tokens that indicate end of sequence (stop at them)
        eos_tokens = {self.eos_token, self.tokenizer.eos_token, "$", "<eos>", "<pad>"}
        # Also handle pad token
        pad_token = self.tokenizer.pad_token
        if pad_token:
            eos_tokens.add(pad_token)
        
        filtered = []
        for token in tokens:
            if token in bos_tokens:
                continue
            if token in eos_tokens:
                break
            filtered.append(token)
        text = self.tokenizer.convert_tokens_to_string(filtered)
        return text.replace(" ", "")
