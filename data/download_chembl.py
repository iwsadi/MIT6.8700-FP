#!/usr/bin/env python
"""
Download and prepare SMILES data for pre-training.

Downloads SMILES from ZINC-250K, MOSES, and GuacaMol (full ChEMBL) datasets,
filters by length, removes duplicates, and saves to a .smi file.

By default, downloads ALL available data (no molecule limit) for maximum
pre-training coverage.

Usage:
    python data/download_chembl.py                    # Full dataset (recommended)
    python data/download_chembl.py --max-length 120   # Filter macromolecules
    python data/download_chembl.py --max-molecules 100000  # Limit size (optional)
"""

import argparse
import os
import urllib.request
from pathlib import Path
from typing import List, Set, Optional
import random


# Dataset URLs - ordered by size (smallest to largest)
DATASET_URLS = {
    # ZINC-250K (commonly used for molecular generation benchmarks) ~250K molecules
    "zinc250k": "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv",
    # MOSES benchmark training set ~1.5M molecules
    "moses": "https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv",
    # GuacaMol benchmark (ChEMBL-derived, large dataset) ~1.6M molecules
    "guacamol": "https://ndownloader.figshare.com/files/13612760",
}

# Approximate molecule counts for progress reporting
DATASET_SIZES = {
    "zinc250k": 250000,
    "moses": 1500000,
    "guacamol": 1600000,
}


def download_file(url: str, output_path: str) -> bool:
    """Download a file from URL with progress reporting."""
    print(f"Downloading: {url}")
    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                print(f"\rProgress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end="", flush=True)
        
        urllib.request.urlretrieve(url, output_path, progress_hook)
        print()  # Newline after progress
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        return False


def parse_zinc250k(filepath: str) -> List[str]:
    """Parse ZINC-250K CSV file (SMILES in first column)."""
    smiles = []
    with open(filepath, 'r') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split(',')
            if parts:
                smi = parts[0].strip().strip('"')
                if smi and smi != 'smiles':
                    smiles.append(smi)
    return smiles


def parse_moses(filepath: str) -> List[str]:
    """Parse MOSES CSV file (SMILES in first column)."""
    smiles = []
    with open(filepath, 'r') as f:
        header = f.readline()  # Skip header
        for line in f:
            smi = line.strip().split(',')[0].strip()
            if smi and smi != 'SMILES':
                smiles.append(smi)
    return smiles


def parse_guacamol(filepath: str) -> List[str]:
    """Parse GuacaMol file (one SMILES per line)."""
    smiles = []
    with open(filepath, 'r') as f:
        for line in f:
            smi = line.strip()
            if smi:
                smiles.append(smi)
    return smiles


def is_valid_smiles_basic(smi: str, max_length: int = 100) -> bool:
    """Basic SMILES validation without RDKit."""
    if not smi or len(smi) > max_length:
        return False
    
    # Must contain at least one carbon or heteroatom
    if not any(c in smi for c in 'CNOSPFIBcnospfib'):
        return False
    
    # Check bracket balance
    if smi.count('[') != smi.count(']'):
        return False
    if smi.count('(') != smi.count(')'):
        return False
    
    return True


def validate_with_rdkit(smiles_list: List[str], max_length: int = 100) -> List[str]:
    """Validate and canonicalize SMILES using RDKit."""
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
        
        valid_smiles = []
        for smi in smiles_list:
            if len(smi) > max_length:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                # Canonicalize
                canonical = Chem.MolToSmiles(mol)
                if len(canonical) <= max_length:
                    valid_smiles.append(canonical)
        return valid_smiles
    except ImportError:
        print("RDKit not available, using basic validation...")
        return [smi for smi in smiles_list if is_valid_smiles_basic(smi, max_length)]


def download_dataset(dataset_name: str, output_dir: Path) -> List[str]:
    """Download and parse a dataset."""
    if dataset_name not in DATASET_URLS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    url = DATASET_URLS[dataset_name]
    temp_file = output_dir / f"{dataset_name}_temp.txt"
    
    if not download_file(url, str(temp_file)):
        return []
    
    # Parse based on dataset format
    if dataset_name == "zinc250k":
        smiles = parse_zinc250k(str(temp_file))
    elif dataset_name == "moses":
        smiles = parse_moses(str(temp_file))
    elif dataset_name == "guacamol":
        smiles = parse_guacamol(str(temp_file))
    else:
        smiles = []
    
    # Clean up temp file
    os.remove(temp_file)
    
    print(f"Parsed {len(smiles)} SMILES from {dataset_name}")
    return smiles


def main():
    parser = argparse.ArgumentParser(
        description="Download FULL ChEMBL/ZINC/MOSES SMILES data for pre-training"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="data/chembl_full.smi",
        help="Output file path"
    )
    parser.add_argument(
        "--max-molecules", "-n", type=int, default=None,
        help="Maximum number of molecules (default: no limit, use all data)"
    )
    parser.add_argument(
        "--max-length", type=int, default=120,
        help="Maximum SMILES length (filter macromolecules, default: 120)"
    )
    parser.add_argument(
        "--min-length", type=int, default=5,
        help="Minimum SMILES length (filter tiny fragments, default: 5)"
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=["zinc250k", "moses", "guacamol"],
        choices=["zinc250k", "moses", "guacamol"],
        help="Datasets to download (default: all three for maximum coverage)"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate SMILES with RDKit (slower but more accurate)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FULL ChEMBL DATASET DOWNLOAD")
    print("=" * 60)
    print(f"Output: {args.output}")
    print(f"Max length: {args.max_length} (filters macromolecules)")
    print(f"Min length: {args.min_length} (filters fragments)")
    print(f"Max molecules: {'No limit (full dataset)' if args.max_molecules is None else args.max_molecules}")
    print(f"Datasets: {args.datasets}")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_smiles: List[str] = []
    
    # Download ALL specified datasets
    for dataset in args.datasets:
        expected_size = DATASET_SIZES.get(dataset, "unknown")
        print(f"\n=== Downloading {dataset} (expected ~{expected_size:,} molecules) ===")
        smiles = download_dataset(dataset, output_path.parent)
        all_smiles.extend(smiles)
        print(f"Running total: {len(all_smiles):,} SMILES")
    
    if not all_smiles:
        print("\nNo SMILES downloaded. Check your internet connection.")
        return
    
    print(f"\n=== Processing {len(all_smiles):,} SMILES ===")
    
    # Remove duplicates
    print("Removing duplicates...")
    unique_smiles: Set[str] = set()
    for smi in all_smiles:
        smi = smi.strip()
        if smi:
            unique_smiles.add(smi)
    print(f"Unique SMILES: {len(unique_smiles):,}")
    
    # Filter by length
    print(f"Filtering by length ({args.min_length} <= len <= {args.max_length})...")
    smiles_list = list(unique_smiles)
    
    if args.validate:
        print("Validating with RDKit (this may take a while for large datasets)...")
        smiles_list = validate_with_rdkit(smiles_list, args.max_length)
        # Also filter by min length
        smiles_list = [s for s in smiles_list if len(s) >= args.min_length]
    else:
        smiles_list = [s for s in smiles_list 
                       if is_valid_smiles_basic(s, args.max_length) and len(s) >= args.min_length]
    
    print(f"Valid SMILES after filtering: {len(smiles_list):,}")
    
    # Shuffle
    print("Shuffling...")
    random.seed(args.seed)
    random.shuffle(smiles_list)
    
    # Optionally limit (but default is no limit)
    if args.max_molecules is not None and len(smiles_list) > args.max_molecules:
        smiles_list = smiles_list[:args.max_molecules]
        print(f"Trimmed to {len(smiles_list):,} molecules")
    
    # Save to file
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        for smi in smiles_list:
            f.write(smi + '\n')
    
    # Calculate statistics
    lengths = [len(s) for s in smiles_list]
    
    print(f"\n" + "=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"Total molecules: {len(smiles_list):,}")
    print(f"Output file: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"\nLength Statistics:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {sum(lengths) / len(lengths):.1f}")
    print(f"  Median: {sorted(lengths)[len(lengths)//2]}")
    
    # Print example command
    print(f"\n" + "=" * 60)
    print("NEXT STEP: Pre-training")
    print("=" * 60)
    print(f"python pretrain_supervised.py \\")
    print(f"    --data {output_path} \\")
    print(f"    --prior pre_trained_models/ChEMBL/random.prior.new \\")
    print(f"    --epochs 5 \\")
    print(f"    --batch-size 128 \\")
    print(f"    --early-stopping")


if __name__ == "__main__":
    main()
