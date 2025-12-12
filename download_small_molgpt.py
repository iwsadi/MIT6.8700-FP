"""
Download small MolGPT model (~7M params) for fast RL training.

Target model: jonghyunlee/MolGPT_pretrained-by-ZINC15
  - ~6.9M parameters
  - 8 layers, 256 hidden, 8 heads
  - 99.68% validity on ZINC15
  - ~12x faster than 87M model

Saves to: pre_trained_models/ChEMBL/molgpt.prior
"""

import os
import sys
from pathlib import Path


def download_small_molgpt():
    print("\n" + "="*60)
    print("  Downloading Small MolGPT (~7M params)")
    print("="*60)
    
    try:
        from transformers import GPT2LMHeadModel, AutoTokenizer, GPT2Tokenizer
    except ImportError:
        print("Installing transformers...")
        os.system(f"{sys.executable} -m pip install transformers")
        from transformers import GPT2LMHeadModel, AutoTokenizer, GPT2Tokenizer
    
    # Model options (smallest to largest)
    models_to_try = [
        ("jonghyunlee/MolGPT_pretrained-by-ZINC15", "~6.9M params, 8 layers"),
        ("jonghyunlee/MolGPT_long_context_pretrained-by-ZINC15", "~6.9M params, long context"),
    ]
    
    save_dir = Path("pre_trained_models/ChEMBL/molgpt_small")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model = None
    tokenizer = None
    model_name = None
    
    for name, desc in models_to_try:
        print(f"\n  Trying: {name}")
        print(f"  Description: {desc}")
        
        try:
            # Try loading model
            print("  Loading model...")
            model = GPT2LMHeadModel.from_pretrained(name)
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            print(f"  ✅ Model loaded: {num_params:,} parameters")
            
            # Try loading tokenizer with multiple methods
            print("  Loading tokenizer...")
            tokenizer = None
            
            tokenizer_methods = [
                ("AutoTokenizer", lambda: AutoTokenizer.from_pretrained(name)),
                ("AutoTokenizer (slow)", lambda: AutoTokenizer.from_pretrained(name, use_fast=False)),
                ("GPT2Tokenizer", lambda: GPT2Tokenizer.from_pretrained(name)),
                ("AutoTokenizer (trust)", lambda: AutoTokenizer.from_pretrained(name, trust_remote_code=True)),
            ]
            
            for method_name, loader in tokenizer_methods:
                try:
                    tokenizer = loader()
                    print(f"  ✅ Tokenizer loaded via {method_name}")
                    break
                except Exception as e:
                    print(f"    ❌ {method_name} failed: {str(e)[:50]}...")
                    continue
            
            if tokenizer is None:
                print("  ⚠️ Could not load tokenizer, trying to create one...")
                # Try to get vocab from model config
                tokenizer = create_smiles_tokenizer(model.config.vocab_size)
            
            model_name = name
            break
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue
    
    if model is None:
        print("\n❌ Could not download any small model!")
        print("Falling back to local 87M model...")
        
        # Use already downloaded model
        local_path = "pre_trained_models/molgpt/entropy_gpt2_zinc_87m"
        if Path(local_path).exists():
            print(f"  Using: {local_path}")
            
            # Create symlink or copy reference
            with open(save_dir / "model_path.txt", "w") as f:
                f.write(local_path)
            
            return local_path
        return None
    
    # Save model and tokenizer
    print(f"\n  Saving to {save_dir}...")
    model.save_pretrained(save_dir)
    
    if tokenizer:
        tokenizer.save_pretrained(save_dir)
    
    # Save model info
    info = {
        "source": model_name,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "vocab_size": model.config.vocab_size,
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "n_embd": model.config.n_embd,
    }
    
    import json
    with open(save_dir / "model_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\n  ✅ Saved to: {save_dir}")
    print(f"  Model info: {info}")
    
    # Also create the molgpt.prior symlink/reference
    prior_path = Path("pre_trained_models/ChEMBL/molgpt.prior")
    with open(prior_path, "w") as f:
        f.write(str(save_dir.absolute()))
    
    print(f"  ✅ Reference created: {prior_path}")
    
    # Test generation
    print("\n  Testing generation...")
    test_generation(model, tokenizer)
    
    return str(save_dir)


def create_smiles_tokenizer(vocab_size):
    """Create a basic tokenizer for SMILES if HF tokenizer fails."""
    from transformers import PreTrainedTokenizerFast
    
    # Basic SMILES tokens
    tokens = ['<pad>', '<s>', '</s>', '<unk>']
    smiles_chars = list("CNOSPFIBcnospb=#-+\\/:@.()[]123456789%")
    tokens.extend(smiles_chars)
    
    # Add more tokens up to vocab_size
    for i in range(len(tokens), vocab_size):
        tokens.append(f"<extra_{i}>")
    
    # Create vocab dict
    vocab = {tok: i for i, tok in enumerate(tokens[:vocab_size])}
    
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    
    tokenizer_backend = Tokenizer(WordLevel(vocab=vocab, unk_token='<unk>'))
    
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        bos_token='<s>',
        eos_token='</s>',
        pad_token='<pad>',
        unk_token='<unk>',
    )
    
    return tokenizer


def test_generation(model, tokenizer):
    """Test that the model can generate SMILES."""
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Start with BOS token
    if tokenizer.bos_token_id is not None:
        input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
    else:
        input_ids = torch.tensor([[0]], device=device)  # Assume 0 is start
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=64,
            num_return_sequences=5,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or 0,
            eos_token_id=tokenizer.eos_token_id or 1,
        )
    
    print("\n  Sample SMILES:")
    for i, seq in enumerate(outputs):
        smiles = tokenizer.decode(seq, skip_special_tokens=True)
        print(f"    {i+1}. {smiles}")
    
    # Test validity
    try:
        from rdkit import Chem
        valid = 0
        for seq in outputs:
            smiles = tokenizer.decode(seq, skip_special_tokens=True)
            if Chem.MolFromSmiles(smiles) is not None:
                valid += 1
        print(f"\n  Validity: {valid}/{len(outputs)} ({100*valid/len(outputs):.0f}%)")
    except ImportError:
        print("  (RDKit not available for validity check)")


if __name__ == "__main__":
    result = download_small_molgpt()
    
    if result:
        print("\n" + "="*60)
        print("  ✅ DONE!")
        print("="*60)
        print(f"\n  Model saved to: {result}")
        print("\n  Update your config to use:")
        print(f'    "transformer_weights": "{result}"')
        print("\n  Or use the reference path:")
        print('    "transformer_weights": "pre_trained_models/ChEMBL/molgpt_small"')
    else:
        print("\n❌ Download failed!")
