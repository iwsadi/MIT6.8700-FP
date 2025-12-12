"""
Download and validate pre-trained MolGPT models for RL training.

Available models (from HuggingFace):
1. entropy/gpt2_zinc_87m - GPT-2 trained on ZINC (87M params, recommended)
2. jonghyunlee/MolGPT_pretrained-by-ZINC15 - MolGPT from ZINC15
3. ncfrey/ChemGPT-1.2B - Large ChemGPT (1.2B params, requires lots of VRAM)

Usage:
    python download_molgpt.py                           # Download default (entropy/gpt2_zinc_87m)
    python download_molgpt.py --model jonghyunlee/MolGPT_pretrained-by-ZINC15
    python download_molgpt.py --list                    # List available models
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


# Available pre-trained models
AVAILABLE_MODELS = {
    "jonghyunlee/MolGPT_pretrained-by-ZINC15": {
        "description": "Original MolGPT (~6M params) trained on ZINC15",
        "params": "~6M",
        "recommended": True,
        "vram": "~500MB",
        "note": "8 layers, 256 hidden, 8 heads - matches the MolGPT paper",
    },
    "entropy/gpt2_zinc_87m": {
        "description": "Large GPT-2 (87M) trained on ZINC dataset",
        "params": "87M",
        "recommended": False,
        "vram": "~2GB",
    },
    "ncfrey/ChemGPT-1.2B": {
        "description": "Large ChemGPT trained on PubChem",
        "params": "1.2B",
        "recommended": False,
        "vram": "~8GB",
    },
    "zjunlp/MolGen-large": {
        "description": "T5-based MolGen (encoder-decoder)",
        "params": "770M",
        "recommended": False,
        "vram": "~6GB",
        "note": "T5-based, may need different handling",
    },
}


def list_models():
    """Print available models."""
    print("\n" + "=" * 70)
    print("  📦 Available Pre-trained Molecular GPT Models")
    print("=" * 70 + "\n")
    
    for name, info in AVAILABLE_MODELS.items():
        rec = " ⭐ RECOMMENDED" if info.get("recommended") else ""
        print(f"  {name}{rec}")
        print(f"     Parameters: {info['params']}")
        print(f"     VRAM: {info['vram']}")
        print(f"     {info['description']}")
        if "note" in info:
            print(f"     ⚠️  {info['note']}")
        print()
    
    print("=" * 70)
    print("  Usage: python download_molgpt.py --model <model_name>")
    print("=" * 70 + "\n")


def download_model(model_name: str, save_dir: str, test_generation: bool = True):
    """
    Download a pre-trained model from HuggingFace and optionally test it.
    
    Args:
        model_name: HuggingFace model identifier
        save_dir: Local directory to save the model
        test_generation: Whether to test molecule generation
    """
    print("\n" + "=" * 70)
    print(f"  🚀 Downloading: {model_name}")
    print("=" * 70 + "\n")
    
    # Install transformers if needed
    try:
        from transformers import GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        print("❌ transformers not installed. Installing...")
        os.system(f"{sys.executable} -m pip install transformers torch")
        from transformers import GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
        import torch
    
    save_path = Path(save_dir) / model_name.replace("/", "_")
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Save location: {save_path}")
    print(f"⏳ Downloading model and tokenizer...")
    
    # Download tokenizer (try multiple methods for compatibility)
    tokenizer = None
    tokenizer_methods = [
        ("AutoTokenizer", lambda: AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)),
        ("AutoTokenizer (slow)", lambda: AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)),
        ("GPT2Tokenizer", lambda: __import__('transformers').GPT2Tokenizer.from_pretrained(model_name)),
        ("PreTrainedTokenizerFast", lambda: __import__('transformers').PreTrainedTokenizerFast.from_pretrained(model_name)),
    ]
    
    for method_name, loader in tokenizer_methods:
        try:
            tokenizer = loader()
            print(f"  ✅ Tokenizer loaded ({method_name}): {len(tokenizer)} tokens")
            break
        except Exception as e:
            print(f"  ⚠️ {method_name} failed: {str(e)[:80]}...")
    
    if tokenizer is None:
        print(f"  ❌ All tokenizer loading methods failed")
        print(f"  💡 This model may have a corrupted tokenizer. Try: entropy/gpt2_zinc_87m instead")
        return None
    
    # Download model
    try:
        # Try GPT2 first, then generic causal LM
        try:
            model = GPT2LMHeadModel.from_pretrained(model_name)
        except:
            model = AutoModelForCausalLM.from_pretrained(model_name)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  ✅ Model loaded: {n_params:,} parameters")
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        return None
    
    # Ensure special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"
    if tokenizer.bos_token is None:
        # Try common alternatives
        if "<s>" in tokenizer.get_vocab():
            tokenizer.bos_token = "<s>"
        elif "[CLS]" in tokenizer.get_vocab():
            tokenizer.bos_token = "[CLS]"
        else:
            tokenizer.add_special_tokens({"bos_token": "<bos>"})
            model.resize_token_embeddings(len(tokenizer))
    
    print(f"\n📋 Model info:")
    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   BOS token: '{tokenizer.bos_token}' (id={tokenizer.bos_token_id})")
    print(f"   EOS token: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")
    print(f"   PAD token: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
    
    # Save locally
    print(f"\n💾 Saving to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"  ✅ Model saved!")
    
    # Test generation
    if test_generation:
        print("\n" + "-" * 70)
        print("  🧪 Testing molecule generation...")
        print("-" * 70 + "\n")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")
        model.to(device)
        model.eval()
        
        # Try to generate a few molecules
        try:
            # Start with BOS token
            bos_id = tokenizer.bos_token_id
            if bos_id is None:
                # Fallback: try encoding a start character
                bos_id = tokenizer.encode("C")[0]
            
            input_ids = torch.tensor([[bos_id]], device=device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_length=100,
                    num_return_sequences=5,
                    do_sample=True,
                    temperature=1.0,
                    top_k=50,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode
            smiles_list = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            smiles_list = [s.replace(" ", "") for s in smiles_list]
            
            print("  Generated SMILES samples:")
            for i, smi in enumerate(smiles_list):
                print(f"    {i+1}. {smi[:80]}{'...' if len(smi) > 80 else ''}")
            
            # Validate with RDKit if available
            try:
                from rdkit import Chem
                valid = sum(1 for s in smiles_list if Chem.MolFromSmiles(s) is not None)
                print(f"\n  📊 Validity: {valid}/{len(smiles_list)} ({100*valid/len(smiles_list):.0f}%)")
            except ImportError:
                print("\n  (RDKit not available for validation)")
                
        except Exception as e:
            print(f"  ⚠️  Generation test failed: {e}")
            print("      This may be normal for some model architectures.")
    
    print("\n" + "=" * 70)
    print("  ✅ DOWNLOAD COMPLETE!")
    print("=" * 70)
    print(f"\n  Model saved to: {save_path}")
    print(f"\n  To use in RL training:")
    print(f"    1. Set 'transformer_weights' in config to: '{save_path}'")
    print(f"    2. Or use HuggingFace ID directly: '{model_name}'")
    print(f"\n  Config example (config_gpt2_rl.json):")
    print(f'    "transformer_weights": "{model_name}"')
    print("=" * 70 + "\n")
    
    return save_path


def verify_with_actor(model_path: str):
    """Verify the model works with ActorModelGPT2."""
    print("\n" + "-" * 70)
    print("  🔬 Verifying with ActorModelGPT2...")
    print("-" * 70 + "\n")
    
    try:
        from smiles_rl.model.actor_model_gpt2 import ActorModelGPT2
        
        actor = ActorModelGPT2.from_pretrained(model_path)
        print(f"  ✅ ActorModelGPT2 loaded successfully!")
        
        # Sample some molecules
        seqs, smiles, log_probs = actor.sample(batch_size=5)
        print(f"\n  Sampled {len(smiles)} molecules:")
        for i, s in enumerate(smiles):
            print(f"    {i+1}. {s[:70]}{'...' if len(s) > 70 else ''}")
        
        # Check validity
        try:
            from rdkit import Chem
            valid = sum(1 for s in smiles if Chem.MolFromSmiles(s) is not None)
            print(f"\n  📊 Validity: {valid}/{len(smiles)} ({100*valid/len(smiles):.0f}%)")
        except ImportError:
            pass
        
        print("\n  ✅ Model is ready for RL training!")
        return True
        
    except Exception as e:
        print(f"  ❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-trained MolGPT models for RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="jonghyunlee/MolGPT_pretrained-by-ZINC15",
        help="HuggingFace model name (default: jonghyunlee/MolGPT_pretrained-by-ZINC15)"
    )
    parser.add_argument(
        "--save-dir", "-o",
        type=str,
        default="pre_trained_models/molgpt",
        help="Directory to save the model (default: pre_trained_models/molgpt)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip generation test"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify model works with ActorModelGPT2"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return
    
    # Download
    save_path = download_model(
        args.model, 
        args.save_dir, 
        test_generation=not args.skip_test
    )
    
    # Verify with ActorModelGPT2
    if save_path and args.verify:
        verify_with_actor(str(save_path))


if __name__ == "__main__":
    main()
