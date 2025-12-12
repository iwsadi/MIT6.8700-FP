"""
Plot training metrics from PPO MolGPT experiments.

Reads CSV files from results folders and generates comparison plots.

Usage:
    python plot_training.py results_ppo_molgpt_small
    python plot_training.py results_ppo_molgpt results_ppo_molgpt_small --compare
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional


def load_metrics(result_folder: str) -> Optional[pd.DataFrame]:
    """Load training metrics from CSV file."""
    csv_path = Path(result_folder) / "training_metrics.csv"
    
    if not csv_path.exists():
        print(f"  ⚠️ No metrics file found at {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"  ✅ Loaded {len(df)} steps from {csv_path}")
    return df


def plot_single_run(df: pd.DataFrame, result_folder: str, save_path: Optional[str] = None):
    """Plot metrics for a single training run."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Training Metrics: {Path(result_folder).name}", fontsize=14)
    
    # 1. Validity
    ax = axes[0, 0]
    ax.plot(df['step'], df['validity'], 'b-', linewidth=1.5)
    ax.axhline(y=100, color='g', linestyle='--', alpha=0.5, label='100%')
    ax.set_xlabel('Step')
    ax.set_ylabel('Validity (%)')
    ax.set_title('SMILES Validity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Scores
    ax = axes[0, 1]
    ax.plot(df['step'], df['raw_score'], 'r-', label='Raw Score', linewidth=1.5)
    ax.plot(df['step'], df['shaped_score'], 'b-', label='Shaped Score', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Score')
    ax.set_title('Activity Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Length with std
    ax = axes[0, 2]
    mean_len = df['mean_length']
    std_len = df['std_length']
    ax.plot(df['step'], mean_len, 'g-', linewidth=1.5)
    ax.fill_between(df['step'], mean_len - std_len, mean_len + std_len, alpha=0.3, color='g')
    ax.set_xlabel('Step')
    ax.set_ylabel('SMILES Length')
    ax.set_title('Sequence Length (mean ± std)')
    ax.grid(True, alpha=0.3)
    
    # 4. Actor Loss
    ax = axes[1, 0]
    ax.plot(df['step'], df['actor_loss'], 'purple', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Actor Loss')
    ax.grid(True, alpha=0.3)
    
    # 5. Critic Loss
    ax = axes[1, 1]
    ax.plot(df['step'], df['critic_loss'], 'orange', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Critic Loss')
    ax.grid(True, alpha=0.3)
    
    # 6. Entropy
    ax = axes[1, 2]
    ax.plot(df['step'], df['entropy'], 'cyan', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Entropy')
    ax.set_title('Policy Entropy')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved plot to {save_path}")
    
    plt.show()


def plot_comparison(dfs: List[pd.DataFrame], names: List[str], save_path: Optional[str] = None):
    """Plot comparison of multiple training runs."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Training Comparison", fontsize=14)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(dfs)))
    
    for df, name, color in zip(dfs, names, colors):
        # 1. Validity
        axes[0, 0].plot(df['step'], df['validity'], color=color, label=name, linewidth=1.5)
        
        # 2. Raw Score
        axes[0, 1].plot(df['step'], df['raw_score'], color=color, label=name, linewidth=1.5)
        
        # 3. Length
        axes[0, 2].plot(df['step'], df['mean_length'], color=color, label=name, linewidth=1.5)
        
        # 4. Actor Loss
        axes[1, 0].plot(df['step'], df['actor_loss'], color=color, label=name, linewidth=1.5)
        
        # 5. Critic Loss
        axes[1, 1].plot(df['step'], df['critic_loss'], color=color, label=name, linewidth=1.5)
        
        # 6. Entropy
        axes[1, 2].plot(df['step'], df['entropy'], color=color, label=name, linewidth=1.5)
    
    titles = ['Validity (%)', 'Raw Score', 'Mean Length', 'Actor Loss', 'Critic Loss', 'Entropy']
    for ax, title in zip(axes.flat, titles):
        ax.set_xlabel('Step')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved comparison plot to {save_path}")
    
    plt.show()


def print_summary(df: pd.DataFrame, name: str):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print(f"  Summary: {name}")
    print(f"{'='*60}")
    print(f"  Total steps: {len(df)}")
    print(f"  Final validity: {df['validity'].iloc[-1]:.1f}%")
    print(f"  Max validity: {df['validity'].max():.1f}%")
    print(f"  Final raw score: {df['raw_score'].iloc[-1]:.4f}")
    print(f"  Max raw score: {df['raw_score'].max():.4f}")
    print(f"  Final mean length: {df['mean_length'].iloc[-1]:.1f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Plot PPO MolGPT training metrics")
    parser.add_argument('folders', nargs='+', help="Result folder(s) to plot")
    parser.add_argument('--compare', action='store_true', help="Compare multiple runs")
    parser.add_argument('--save', type=str, default=None, help="Save plot to file")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  📊 PPO MolGPT Training Plotter")
    print("="*60 + "\n")
    
    # Load data
    dfs = []
    names = []
    
    for folder in args.folders:
        df = load_metrics(folder)
        if df is not None:
            dfs.append(df)
            names.append(Path(folder).name)
            print_summary(df, folder)
    
    if not dfs:
        print("\n❌ No data found to plot!")
        return
    
    # Plot
    if args.compare and len(dfs) > 1:
        plot_comparison(dfs, names, save_path=args.save)
    else:
        for df, folder in zip(dfs, args.folders):
            save_path = args.save if len(dfs) == 1 else f"{folder}_plot.png"
            plot_single_run(df, folder, save_path=save_path)


if __name__ == "__main__":
    main()
