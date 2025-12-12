#!/usr/bin/env python
"""
Supervised Pre-training for Transformer SMILES Generator.

Trains ActorModelTransformer on a corpus of valid SMILES strings using
next-token prediction (CrossEntropyLoss) before RL fine-tuning.

This teaches the model valid SMILES syntax, which is critical for achieving
non-zero validity rates during RL training.

Usage:
    python pretrain_supervised.py --data data/smiles_corpus.smi --prior pre_trained_models/ChEMBL/random.prior.new
    python pretrain_supervised.py --data data/smiles_corpus.smi --prior pre_trained_models/ChEMBL/random.prior.new --epochs 10 --batch-size 128
"""

import argparse
import csv
import json
import sys
import time
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from smiles_rl.model.actor_model_transformer import ActorModelTransformer
from smiles_rl.model.vocabulary import SMILESTokenizer, Vocabulary
from data.dataset import SMILESDataset, load_smiles_from_file


class TrainingHistory:
    """Track and save training metrics for plotting."""
    
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Per-epoch metrics
        self.epochs: List[int] = []
        self.train_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_losses: List[float] = []
        self.val_accs: List[float] = []
        self.validity_rates: List[float] = []
        self.learning_rates: List[float] = []
        self.epoch_times: List[float] = []
        
        # Best metrics
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def add_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        validity: float,
        lr: float,
        epoch_time: float,
    ):
        """Record metrics for one epoch."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        self.validity_rates.append(validity)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
        
        # Track best
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
    
    def save_csv(self, filename: str = "training_history.csv"):
        """Save metrics to CSV file."""
        csv_path = self.save_dir / filename
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
                'validity_rate', 'learning_rate', 'epoch_time'
            ])
            
            for i in range(len(self.epochs)):
                writer.writerow([
                    self.epochs[i],
                    f"{self.train_losses[i]:.6f}",
                    f"{self.train_accs[i]:.2f}",
                    f"{self.val_losses[i]:.6f}",
                    f"{self.val_accs[i]:.2f}",
                    f"{self.validity_rates[i]:.4f}",
                    f"{self.learning_rates[i]:.2e}",
                    f"{self.epoch_times[i]:.1f}",
                ])
        
        debug_print(f"[DEBUG] Training history saved to: {csv_path}")
        return csv_path
    
    def save_json(self, filename: str = "training_history.json"):
        """Save metrics to JSON file."""
        json_path = self.save_dir / filename
        
        data = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'validity_rates': self.validity_rates,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'total_time': sum(self.epoch_times),
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        debug_print(f"[DEBUG] Training history saved to: {json_path}")
        return json_path
    
    def plot(self, filename: str = "training_plot.png"):
        """Generate and save training plots."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot 1: Loss curves
            ax1 = axes[0, 0]
            ax1.plot(self.epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
            ax1.plot(self.epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
            ax1.axvline(x=self.best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best (epoch {self.best_epoch})')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training & Validation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Accuracy curves
            ax2 = axes[0, 1]
            ax2.plot(self.epochs, self.train_accs, 'b-', label='Train Acc', linewidth=2)
            ax2.plot(self.epochs, self.val_accs, 'r-', label='Val Acc', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Training & Validation Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Validity rate
            ax3 = axes[1, 0]
            ax3.plot(self.epochs, [v * 100 for v in self.validity_rates], 'g-', linewidth=2, marker='o')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Validity Rate (%)')
            ax3.set_title('SMILES Validity Rate')
            ax3.set_ylim(0, 105)
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Learning rate
            ax4 = axes[1, 1]
            ax4.plot(self.epochs, self.learning_rates, 'm-', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_title('Learning Rate Schedule')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_path = self.save_dir / filename
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            debug_print(f"[DEBUG] Training plot saved to: {plot_path}")
            return plot_path
            
        except ImportError:
            debug_print("[WARNING] matplotlib not available, skipping plot generation")
            return None
    
    def print_summary(self):
        """Print training summary."""
        if not self.epochs:
            return
        
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Total epochs: {len(self.epochs)}")
        print(f"Total time: {sum(self.epoch_times):.1f}s ({sum(self.epoch_times)/60:.1f} min)")
        print(f"\nBest model (epoch {self.best_epoch}):")
        print(f"  Val Loss: {self.best_val_loss:.4f}")
        if self.best_epoch > 0:
            idx = self.best_epoch - 1
            print(f"  Val Acc: {self.val_accs[idx]:.1f}%")
            print(f"  Validity: {self.validity_rates[idx]*100:.1f}%")
        print(f"\nFinal (epoch {self.epochs[-1]}):")
        print(f"  Train Loss: {self.train_losses[-1]:.4f}")
        print(f"  Val Loss: {self.val_losses[-1]:.4f}")
        print(f"  Val Acc: {self.val_accs[-1]:.1f}%")
        print(f"  Validity: {self.validity_rates[-1]*100:.1f}%")
        print("=" * 60)


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    
    Args:
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change in validation loss to qualify as improvement
        verbose: Whether to print messages
    """
    
    def __init__(self, patience: int = 2, min_delta: float = 0.01, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, val_loss: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        # Check if loss improved by at least min_delta
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            if self.verbose:
                improvement = self.best_loss - val_loss
                debug_print(f"[EarlyStopping] Val loss improved by {improvement:.4f} "
                           f"({self.best_loss:.4f} -> {val_loss:.4f})")
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            self.should_stop = False
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                debug_print(f"[EarlyStopping] No improvement for {self.counter}/{self.patience} epochs "
                           f"(best: {self.best_loss:.4f}, current: {val_loss:.4f}, "
                           f"min_delta: {self.min_delta})")
            
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    debug_print(f"[EarlyStopping] STOPPING! No improvement for {self.patience} consecutive epochs")
        
        return self.should_stop
    
    def reset(self):
        """Reset the early stopping counter."""
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False


class TrainingVisualizer:
    """
    Generate visualizations and plots during training.
    
    Creates:
    - Loss curve plot (updated each epoch)
    - Validity progress plot (updated each epoch)
    - Sample molecule grid images (per epoch)
    - Text log of generated SMILES
    """
    
    def __init__(self, save_dir: Path):
        """Initialize visualizer with output directory."""
        self.save_dir = Path(save_dir) / "visualizations"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics history
        self.epochs: List[int] = []
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.validity_rates: List[float] = []
        self.accuracies: List[float] = []
        
        # Sample log file
        self.samples_file = self.save_dir / "samples.txt"
        # Initialize samples file with header
        with open(self.samples_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SMILES SAMPLES LOG - Pre-training Progress\n")
            f.write("=" * 60 + "\n\n")
        
        print(f"[Visualizer] Output directory: {self.save_dir}", flush=True)
    
    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_acc: float,
        model: 'ActorModelTransformer',
        n_samples: int = 64,
    ):
        """
        Update visualizations after an epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss for this epoch
            val_loss: Validation loss for this epoch
            val_acc: Validation accuracy
            model: The model to sample from
            n_samples: Number of SMILES to sample for validity check
        """
        # Record metrics
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.accuracies.append(val_acc)
        
        # Sample molecules and calculate validity
        print(f"[Visualizer] Sampling {n_samples} molecules...", flush=True)
        samples, valid_smiles, validity = self._sample_and_validate(model, n_samples)
        self.validity_rates.append(validity)
        
        # Generate all visualizations
        self._plot_loss_curves()
        self._plot_validity_curve()
        self._save_molecule_grid(epoch, valid_smiles)
        self._log_samples(epoch, samples, validity)
        
        print(f"[Visualizer] Epoch {epoch} visualizations saved", flush=True)
        
        return samples, validity
    
    def _sample_and_validate(
        self, 
        model: 'ActorModelTransformer', 
        n_samples: int
    ) -> Tuple[List[str], List[str], float]:
        """
        Sample SMILES from the model and validate with RDKit.
        
        Uses small batches and clears GPU cache to avoid OOM errors.
        
        Returns:
            all_samples: All generated SMILES
            valid_smiles: Only valid SMILES
            validity_rate: Fraction of valid SMILES (0-1)
        """
        model.transformer.eval()
        
        # Clear GPU cache before sampling
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        smiles_list = []
        batch_size = min(8, n_samples)  # Small batches to avoid OOM
        
        try:
            with torch.no_grad():
                remaining = n_samples
                while remaining > 0:
                    current_batch = min(batch_size, remaining)
                    try:
                        _, batch_smiles, _ = model.sample(batch_size=current_batch)
                        smiles_list.extend(batch_smiles)
                        remaining -= current_batch
                        
                        # Clear cache between batches
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"[Visualizer] OOM, reducing batch size...", flush=True)
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            batch_size = max(1, batch_size // 2)
                            if batch_size < 1:
                                break
                        else:
                            raise
        except Exception as e:
            print(f"[Visualizer] Sampling failed: {e}", flush=True)
            return [], [], 0.0
        
        # Validate with RDKit
        valid_smiles = []
        try:
            from rdkit import Chem
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')
            
            for smi in smiles_list:
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        valid_smiles.append(smi)
                except:
                    continue
            
            validity = len(valid_smiles) / len(smiles_list) if smiles_list else 0.0
            
        except ImportError:
            # Fallback: basic bracket balance check
            for smi in smiles_list:
                if smi and smi.count('[') == smi.count(']') and smi.count('(') == smi.count(')'):
                    valid_smiles.append(smi)
            validity = len(valid_smiles) / len(smiles_list) if smiles_list else 0.0
        
        model.transformer.train()
        return smiles_list, valid_smiles, validity
    
    def _plot_loss_curves(self):
        """Plot training and validation loss curves."""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(self.epochs, self.train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
            ax.plot(self.epochs, self.val_losses, 'r-o', label='Val Loss', linewidth=2, markersize=6)
            
            # Mark best epoch
            if self.val_losses:
                best_idx = self.val_losses.index(min(self.val_losses))
                ax.axvline(x=self.epochs[best_idx], color='g', linestyle='--', alpha=0.7, 
                          label=f'Best (epoch {self.epochs[best_idx]})')
                ax.scatter([self.epochs[best_idx]], [self.val_losses[best_idx]], 
                          color='g', s=100, zorder=5)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Training Progress: Loss Curves', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Set integer x-axis ticks
            ax.set_xticks(self.epochs)
            
            plt.tight_layout()
            plt.savefig(self.save_dir / "loss_curve.png", dpi=150, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            print("[Visualizer] matplotlib not available, skipping loss plot", flush=True)
        except Exception as e:
            print(f"[Visualizer] Error plotting loss curves: {e}", flush=True)
    
    def _plot_validity_curve(self):
        """Plot SMILES validity rate over epochs."""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            validity_pct = [v * 100 for v in self.validity_rates]
            
            ax.plot(self.epochs, validity_pct, 'g-o', linewidth=2, markersize=8)
            ax.fill_between(self.epochs, 0, validity_pct, alpha=0.3, color='green')
            
            # Add value labels
            for i, (ep, val) in enumerate(zip(self.epochs, validity_pct)):
                ax.annotate(f'{val:.1f}%', (ep, val), textcoords="offset points", 
                           xytext=(0, 10), ha='center', fontsize=9)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Valid SMILES (%)', fontsize=12)
            ax.set_title('SMILES Validity Progress (Model Learning Chemistry)', 
                        fontsize=14, fontweight='bold')
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(self.epochs)
            
            # Add target line at 90%
            ax.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='Target (90%)')
            ax.legend(fontsize=10)
            
            plt.tight_layout()
            plt.savefig(self.save_dir / "validity_curve.png", dpi=150, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            print("[Visualizer] matplotlib not available, skipping validity plot", flush=True)
        except Exception as e:
            print(f"[Visualizer] Error plotting validity curve: {e}", flush=True)
    
    def _save_molecule_grid(self, epoch: int, valid_smiles: List[str], n_mols: int = 9):
        """Create and save a grid image of sample molecules."""
        if not valid_smiles:
            print(f"[Visualizer] No valid molecules to display for epoch {epoch}", flush=True)
            return
        
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')
            
            # Take first n_mols valid molecules
            mols = []
            legends = []
            for i, smi in enumerate(valid_smiles[:n_mols]):
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        mols.append(mol)
                        # Truncate long SMILES for legend
                        legend = smi[:30] + "..." if len(smi) > 30 else smi
                        legends.append(legend)
                except:
                    continue
            
            if not mols:
                print(f"[Visualizer] No valid mols for grid at epoch {epoch}", flush=True)
                return
            
            # Create grid image
            n_cols = 3
            n_rows = (len(mols) + n_cols - 1) // n_cols
            
            img = Draw.MolsToGridImage(
                mols, 
                molsPerRow=n_cols, 
                subImgSize=(300, 300),
                legends=legends,
                returnPNG=False
            )
            
            # Save image
            grid_path = self.save_dir / f"epoch_{epoch}_samples.png"
            img.save(grid_path)
            print(f"[Visualizer] Molecule grid saved: {grid_path}", flush=True)
            
        except ImportError:
            print("[Visualizer] RDKit Draw not available, skipping molecule grid", flush=True)
        except Exception as e:
            print(f"[Visualizer] Error creating molecule grid: {e}", flush=True)
    
    def _log_samples(self, epoch: int, samples: List[str], validity: float):
        """Append generated SMILES to the samples log file."""
        try:
            with open(self.samples_file, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"EPOCH {epoch} - Validity: {validity*100:.1f}%\n")
                f.write(f"{'='*60}\n")
                f.write(f"Total samples: {len(samples)}\n")
                f.write(f"Valid samples: {int(validity * len(samples))}\n\n")
                
                f.write("Sample SMILES:\n")
                f.write("-" * 40 + "\n")
                for i, smi in enumerate(samples[:20]):  # Log first 20
                    # Mark valid/invalid
                    try:
                        from rdkit import Chem
                        mol = Chem.MolFromSmiles(smi)
                        status = "✓" if mol is not None else "✗"
                    except:
                        status = "?"
                    f.write(f"{i+1:3d}. [{status}] {smi}\n")
                
                if len(samples) > 20:
                    f.write(f"... and {len(samples) - 20} more samples\n")
                f.write("\n")
                
        except Exception as e:
            print(f"[Visualizer] Error logging samples: {e}", flush=True)
    
    def create_summary_plot(self):
        """Create a final summary plot with all metrics."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot 1: Loss curves
            ax1 = axes[0, 0]
            ax1.plot(self.epochs, self.train_losses, 'b-o', label='Train Loss', linewidth=2)
            ax1.plot(self.epochs, self.val_losses, 'r-o', label='Val Loss', linewidth=2)
            if self.val_losses:
                best_idx = self.val_losses.index(min(self.val_losses))
                ax1.axvline(x=self.epochs[best_idx], color='g', linestyle='--', alpha=0.7)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Loss Curves')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Validity curve
            ax2 = axes[0, 1]
            validity_pct = [v * 100 for v in self.validity_rates]
            ax2.plot(self.epochs, validity_pct, 'g-o', linewidth=2, markersize=8)
            ax2.fill_between(self.epochs, 0, validity_pct, alpha=0.3, color='green')
            ax2.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='Target 90%')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Valid SMILES (%)')
            ax2.set_title('SMILES Validity Progress')
            ax2.set_ylim(0, 105)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Accuracy curve
            ax3 = axes[1, 0]
            ax3.plot(self.epochs, self.accuracies, 'm-o', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Accuracy (%)')
            ax3.set_title('Token Prediction Accuracy')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Summary text
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            if self.epochs:
                best_val_idx = self.val_losses.index(min(self.val_losses))
                final_idx = -1
                
                summary_text = f"""
TRAINING SUMMARY
{'='*40}

Epochs Completed: {len(self.epochs)}

Best Model (Epoch {self.epochs[best_val_idx]}):
  • Val Loss: {self.val_losses[best_val_idx]:.4f}
  • Validity: {self.validity_rates[best_val_idx]*100:.1f}%
  • Accuracy: {self.accuracies[best_val_idx]:.1f}%

Final Model (Epoch {self.epochs[final_idx]}):
  • Train Loss: {self.train_losses[final_idx]:.4f}
  • Val Loss: {self.val_losses[final_idx]:.4f}
  • Validity: {self.validity_rates[final_idx]*100:.1f}%
  • Accuracy: {self.accuracies[final_idx]:.1f}%

Improvement:
  • Loss: {self.val_losses[0]:.4f} → {self.val_losses[final_idx]:.4f}
  • Validity: {self.validity_rates[0]*100:.1f}% → {self.validity_rates[final_idx]*100:.1f}%
"""
                ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            
            plt.suptitle('Pre-training Summary Report', fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(self.save_dir / "training_summary.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[Visualizer] Summary plot saved: {self.save_dir / 'training_summary.png'}", flush=True)
            
        except ImportError:
            print("[Visualizer] matplotlib not available, skipping summary plot", flush=True)
        except Exception as e:
            print(f"[Visualizer] Error creating summary plot: {e}", flush=True)


def debug_print(msg: str):
    """Print with flush to ensure immediate output."""
    print(msg, flush=True)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal (autoregressive) attention mask.
    
    This is CRITICAL for decoder-only transformers!
    
    The mask ensures that position i can only attend to positions <= i.
    Without this mask, the model can "cheat" by looking at future tokens
    during training, but can't do this during generation.
    
    This mismatch (exposure bias) causes models to:
    - Have high teacher-forced accuracy (sees all tokens)
    - But low real validity (can't see future during generation)
    
    Shape: (1, seq_len, seq_len)
    Value: 1 where attention is allowed, 0 where it should be blocked
    
    Example for seq_len=4:
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]
    """
    # Create lower triangular matrix (1s below and on diagonal, 0s above)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    # Add batch dimension
    return mask.unsqueeze(0)  # (1, seq_len, seq_len)


def create_padding_mask(inputs: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create a padding mask to ignore padding tokens.
    
    Shape: (batch_size, 1, seq_len)
    Value: 1 for real tokens, 0 for padding tokens
    """
    # inputs shape: (batch_size, seq_len)
    mask = (inputs != pad_idx).unsqueeze(1)  # (batch_size, 1, seq_len)
    return mask.float()


def create_combined_mask(inputs: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create combined causal + padding mask for autoregressive training.
    
    This mask:
    1. Prevents attending to future tokens (causal)
    2. Prevents attending to padding tokens
    
    Shape: (batch_size, seq_len, seq_len)
    """
    batch_size, seq_len = inputs.shape
    device = inputs.device
    
    # Causal mask: (1, seq_len, seq_len)
    causal_mask = create_causal_mask(seq_len, device)
    
    # Padding mask: (batch_size, 1, seq_len)
    padding_mask = create_padding_mask(inputs, pad_idx)
    
    # Combine: causal_mask AND padding_mask
    # Broadcasting: (1, seq_len, seq_len) * (batch_size, 1, seq_len)
    # Result: (batch_size, seq_len, seq_len)
    combined_mask = causal_mask * padding_mask
    
    return combined_mask


def load_vocab(prior_path: str) -> Tuple[Vocabulary, SMILESTokenizer, int]:
    """
    Robustly load vocabulary from a prior model checkpoint.
    
    Handles multiple checkpoint formats:
    - REINVENT-style: {'vocabulary': ..., 'tokenizer': ..., 'max_sequence_length': ...}
    - Nested format: {'network': {...}, 'vocabulary': ...}
    - Direct vocabulary object
    
    Args:
        prior_path: Path to the prior model file
        
    Returns:
        vocabulary, tokenizer, max_sequence_length
    """
    debug_print(f"[DEBUG] load_vocab() called with path: {prior_path}")
    
    # Check file exists
    if not os.path.exists(prior_path):
        raise FileNotFoundError(f"Prior file not found: {prior_path}")
    
    file_size = os.path.getsize(prior_path)
    debug_print(f"[DEBUG] File exists, size: {file_size / 1024 / 1024:.2f} MB")
    
    # Try loading as PyTorch checkpoint
    debug_print("[DEBUG] Attempting torch.load()...")
    try:
        # Always load to CPU first to avoid GPU memory issues
        ckpt = torch.load(prior_path, map_location='cpu')
        debug_print(f"[DEBUG] torch.load() succeeded")
        debug_print(f"[DEBUG] Loaded object type: {type(ckpt)}")
    except Exception as e:
        debug_print(f"[DEBUG] torch.load() failed: {e}")
        raise RuntimeError(f"Failed to load prior file: {e}")
    
    # Initialize defaults
    vocabulary = None
    tokenizer = None
    max_seq_len = 256
    
    # Case 1: Dictionary checkpoint
    if isinstance(ckpt, dict):
        debug_print(f"[DEBUG] Checkpoint is a dict with keys: {list(ckpt.keys())}")
        
        # Extract vocabulary
        if 'vocabulary' in ckpt:
            vocabulary = ckpt['vocabulary']
            debug_print(f"[DEBUG] Found 'vocabulary' key, type: {type(vocabulary)}")
        elif 'vocab' in ckpt:
            vocabulary = ckpt['vocab']
            debug_print(f"[DEBUG] Found 'vocab' key, type: {type(vocabulary)}")
        elif 'network' in ckpt and isinstance(ckpt['network'], dict):
            # Check if vocabulary is nested inside network
            if 'vocabulary' in ckpt['network']:
                vocabulary = ckpt['network']['vocabulary']
                debug_print(f"[DEBUG] Found vocabulary nested in 'network'")
        
        # Extract tokenizer
        if 'tokenizer' in ckpt:
            tokenizer = ckpt['tokenizer']
            debug_print(f"[DEBUG] Found 'tokenizer' key, type: {type(tokenizer)}")
        
        # Extract max_sequence_length
        if 'max_sequence_length' in ckpt:
            max_seq_len = ckpt['max_sequence_length']
            debug_print(f"[DEBUG] Found 'max_sequence_length': {max_seq_len}")
        elif 'max_seq_len' in ckpt:
            max_seq_len = ckpt['max_seq_len']
            debug_print(f"[DEBUG] Found 'max_seq_len': {max_seq_len}")
    
    # Case 2: Direct Vocabulary object
    elif isinstance(ckpt, Vocabulary):
        debug_print("[DEBUG] Checkpoint is a direct Vocabulary object")
        vocabulary = ckpt
    
    # Case 3: Unknown format
    else:
        debug_print(f"[DEBUG] Unknown checkpoint format: {type(ckpt)}")
        # Try to use it as vocabulary directly
        vocabulary = ckpt
    
    # Validate vocabulary
    if vocabulary is None:
        raise ValueError("Could not extract vocabulary from checkpoint")
    
    debug_print(f"[DEBUG] Vocabulary type: {type(vocabulary)}")
    
    # Check vocabulary is valid
    try:
        vocab_size = len(vocabulary)
        debug_print(f"[DEBUG] Vocabulary size: {vocab_size}")
    except Exception as e:
        raise ValueError(f"Invalid vocabulary object (can't get length): {e}")
    
    # Check for required tokens
    try:
        start_idx = vocabulary['^']
        end_idx = vocabulary['$']
        debug_print(f"[DEBUG] Start token '^' = {start_idx}")
        debug_print(f"[DEBUG] End token '$' = {end_idx}")
    except KeyError as e:
        raise ValueError(f"Vocabulary missing required token: {e}")
    
    # Create default tokenizer if not found
    if tokenizer is None:
        debug_print("[DEBUG] Creating default SMILESTokenizer")
        tokenizer = SMILESTokenizer()
    
    debug_print("[DEBUG] load_vocab() completed successfully")
    return vocabulary, tokenizer, max_seq_len


def create_model(
    vocabulary: Vocabulary,
    tokenizer: SMILESTokenizer,
    max_seq_len: int,
    network_params: dict,
    device: torch.device,
) -> ActorModelTransformer:
    """Create and initialize the ActorModelTransformer with verbose logging."""
    
    debug_print(f"[DEBUG] create_model() called")
    debug_print(f"[DEBUG]   vocabulary size: {len(vocabulary)}")
    debug_print(f"[DEBUG]   max_seq_len: {max_seq_len}")
    debug_print(f"[DEBUG]   network_params: {network_params}")
    debug_print(f"[DEBUG]   device: {device}")
    
    # Determine if we should use CUDA
    use_cuda = (device.type == 'cuda')
    no_cuda = not use_cuda
    debug_print(f"[DEBUG]   no_cuda flag: {no_cuda}")
    
    debug_print("[DEBUG] Initializing ActorModelTransformer...")
    try:
        model = ActorModelTransformer(
            vocabulary=vocabulary,
            tokenizer=tokenizer,
            network_params=network_params,
            max_sequence_length=max_seq_len,
            no_cuda=no_cuda,
        )
        debug_print("[DEBUG] ActorModelTransformer initialized successfully")
    except Exception as e:
        debug_print(f"[DEBUG] ActorModelTransformer initialization FAILED: {e}")
        raise
    
    # Count parameters
    debug_print("[DEBUG] Counting trainable parameters...")
    n_params = sum(p.numel() for p in model.transformer.parameters() if p.requires_grad)
    debug_print(f"[DEBUG] Trainable parameters: {n_params:,}")
    
    # Verify model is on correct device
    try:
        param_device = next(model.transformer.parameters()).device
        debug_print(f"[DEBUG] Model is on device: {param_device}")
    except StopIteration:
        debug_print("[DEBUG] WARNING: Model has no parameters!")
    
    debug_print("[DEBUG] create_model() completed successfully")
    return model


def train_epoch(
    model: ActorModelTransformer,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int = 50,
    pad_idx: int = 0,
) -> Tuple[float, float]:
    """
    Train for one epoch with CAUSAL MASKING.
    
    IMPORTANT: We use a causal mask to ensure the model can only attend
    to past tokens (not future). This matches the autoregressive generation
    process and addresses the exposure bias problem.
    
    Without causal masking:
    - Model sees all tokens during training (cheating!)
    - But can only see past tokens during generation
    - Result: High training accuracy, low generation validity
    
    Returns:
        avg_loss: Average loss over the epoch
        accuracy: Token prediction accuracy
    """
    model.transformer.train()
    
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Create CAUSAL MASK - CRITICAL for autoregressive training!
        # This prevents the model from "cheating" by looking at future tokens
        causal_mask = create_combined_mask(inputs, pad_idx=pad_idx)
        
        # Forward pass WITH causal mask
        # transformer(inputs, mask) returns logits of shape (batch, seq_len, vocab_size)
        logits = model.transformer(inputs, causal_mask)
        
        # Reshape for loss computation
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)  # (batch * seq_len, vocab_size)
        targets_flat = targets.reshape(-1)            # (batch * seq_len,)
        
        # Compute loss (ignore_index=0 masks out padding tokens)
        loss = criterion(logits_flat, targets_flat)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.transformer.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Track statistics
        total_loss += loss.item() * batch_size
        
        # Calculate accuracy (ignoring padding, but INCLUDING EOS!)
        predictions = logits_flat.argmax(dim=-1)
        # CRITICAL FIX: Use != -100 (not != 0) because padding is now -100
        # EOS (index 0) should be COUNTED in accuracy, not ignored!
        mask = targets_flat != -100  # Non-padding tokens (includes EOS now!)
        total_correct += (predictions[mask] == targets_flat[mask]).sum().item()
        total_tokens += mask.sum().item()
        
        # Log progress
        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            curr_loss = total_loss / ((batch_idx + 1) * batch_size)
            curr_acc = total_correct / total_tokens * 100 if total_tokens > 0 else 0
            print(f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | Avg Loss: {curr_loss:.4f} | "
                  f"Acc: {curr_acc:.1f}% | Time: {elapsed:.1f}s")
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / total_tokens * 100 if total_tokens > 0 else 0
    
    return avg_loss, accuracy


def validate_teacher_forced(
    model: ActorModelTransformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    pad_idx: int = 0,
) -> Tuple[float, float]:
    """
    Validate model using teacher forcing with CAUSAL MASKING.
    
    Even though this is "teacher-forced" (we provide ground truth inputs),
    we still use causal masking to match the training setup.
    
    NOTE: This still gives somewhat optimistic metrics because:
    - The model gets the correct previous tokens as input
    - During real generation, it uses its own (potentially wrong) predictions
    
    Returns:
        avg_loss: Average loss
        accuracy: Token prediction accuracy (may be misleadingly high!)
    """
    model.transformer.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Create causal mask (same as training)
            causal_mask = create_combined_mask(inputs, pad_idx=pad_idx)
            
            # Forward pass WITH causal mask
            logits = model.transformer(inputs, causal_mask)
            
            # Reshape
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.reshape(-1, vocab_size)
            targets_flat = targets.reshape(-1)
            
            # Loss
            loss = criterion(logits_flat, targets_flat)
            total_loss += loss.item() * batch_size
            
            # Accuracy (ignoring padding, but INCLUDING EOS!)
            predictions = logits_flat.argmax(dim=-1)
            # CRITICAL FIX: Use != -100 (not != 0) because padding is now -100
            mask = targets_flat != -100  # Non-padding tokens (includes EOS now!)
            total_correct += (predictions[mask] == targets_flat[mask]).sum().item()
            total_tokens += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / total_tokens * 100 if total_tokens > 0 else 0
    
    return avg_loss, accuracy


def validate_with_generation(
    model: ActorModelTransformer,
    n_samples: int = 100,
    temperature: float = 1.0,
) -> Tuple[List[str], List[str], float, Dict[str, Any]]:
    """
    REAL validation using autoregressive generation (no teacher forcing).
    
    This addresses the EXPOSURE BIAS problem by generating molecules
    from scratch, just like during RL inference. Errors propagate
    naturally, giving us the TRUE validity rate.
    
    Args:
        model: The transformer model
        n_samples: Number of molecules to generate
        temperature: Sampling temperature (1.0=standard, 0.8=conservative)
    
    Returns:
        all_samples: All generated SMILES strings
        valid_samples: Only the valid SMILES strings
        validity_rate: Fraction of valid molecules (0-1)
        stats: Dictionary with detailed statistics
    """
    model.transformer.eval()
    
    all_samples = []
    valid_samples = []
    stats = {
        'total_generated': 0,
        'valid_count': 0,
        'invalid_count': 0,
        'empty_count': 0,
        'avg_length': 0,
        'unique_count': 0,
        'temperature': temperature,
    }
    
    try:
        # CRITICAL: Clear GPU memory before sampling to avoid OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            debug_print(f"[GenVal] Cleared CUDA cache")
        
        # Generate sequences autoregressively from scratch (start token ^)
        # This mimics real RL inference - no ground truth provided
        debug_print(f"[GenVal] Generating {n_samples} molecules (temp={temperature})...")
        
        # Use smaller batch sizes to avoid OOM during generation
        # Generation is memory-intensive due to autoregressive loop
        batch_size_for_sampling = min(16, n_samples)  # Small batches to avoid OOM
        smiles_list = []
        
        with torch.no_grad():  # Prevent gradient tracking during sampling
            remaining = n_samples
            while remaining > 0:
                current_batch = min(batch_size_for_sampling, remaining)
                try:
                    _, batch_smiles, _ = model.sample(batch_size=current_batch)
                    smiles_list.extend(batch_smiles)
                    remaining -= current_batch
                    
                    # Clear cache between batches
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        debug_print(f"[GenVal] OOM with batch_size={current_batch}, trying smaller...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # Try with even smaller batch
                        batch_size_for_sampling = max(1, batch_size_for_sampling // 2)
                        if batch_size_for_sampling < 1:
                            raise
                    else:
                        raise
        
        all_samples = smiles_list
        stats['total_generated'] = len(smiles_list)
        
        # Calculate average length
        lengths = [len(s) for s in smiles_list if s]
        stats['avg_length'] = sum(lengths) / len(lengths) if lengths else 0
        
        # Validate with RDKit (the TRUE test!)
        try:
            from rdkit import Chem
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')
            
            for smi in smiles_list:
                if not smi or len(smi.strip()) == 0:
                    stats['empty_count'] += 1
                    continue
                    
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        valid_samples.append(smi)
                        stats['valid_count'] += 1
                    else:
                        stats['invalid_count'] += 1
                except:
                    stats['invalid_count'] += 1
            
        except ImportError:
            # Fallback: basic bracket balance check
            debug_print("[GenVal] WARNING: RDKit not available, using basic validation")
            for smi in smiles_list:
                if not smi:
                    stats['empty_count'] += 1
                    continue
                if smi.count('[') == smi.count(']') and smi.count('(') == smi.count(')'):
                    valid_samples.append(smi)
                    stats['valid_count'] += 1
                else:
                    stats['invalid_count'] += 1
        
        # Calculate uniqueness
        stats['unique_count'] = len(set(valid_samples))
        
        # Calculate validity rate
        validity_rate = stats['valid_count'] / stats['total_generated'] if stats['total_generated'] > 0 else 0.0
        
        return all_samples, valid_samples, validity_rate, stats
        
    except Exception as e:
        debug_print(f"[GenVal] ERROR during generation: {e}")
        import traceback
        traceback.print_exc()
        return [], [], 0.0, stats


def validate_complete(
    model: ActorModelTransformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    n_gen_samples: int = 100,
    temperature: float = 1.0,
    pad_idx: int = 0,
) -> Dict[str, Any]:
    """
    Complete validation with both teacher-forced AND generative metrics.
    
    This gives you the full picture:
    - Teacher-forced metrics: How well the model fits the data (with causal mask)
    - Generative metrics: How well the model ACTUALLY generates valid molecules
    
    The REAL VALIDITY is what matters for RL!
    
    Args:
        model: The transformer model
        dataloader: Validation data loader
        criterion: Loss function
        device: Computing device
        n_gen_samples: Number of molecules to generate for validity check
        temperature: Sampling temperature
        pad_idx: Padding token index for mask creation
    
    Returns:
        Dictionary containing all validation metrics
    """
    results = {}
    
    # 1. Teacher-Forced Validation (standard but optimistic)
    debug_print("[Validation] Running teacher-forced validation...")
    tf_loss, tf_acc = validate_teacher_forced(model, dataloader, criterion, device, pad_idx=pad_idx)
    results['teacher_forced'] = {
        'loss': tf_loss,
        'accuracy': tf_acc,
    }
    
    # 2. Generative Validation (TRUE performance!)
    debug_print(f"[Validation] Running generative validation ({n_gen_samples} samples)...")
    all_samples, valid_samples, validity_rate, gen_stats = validate_with_generation(
        model, n_samples=n_gen_samples, temperature=temperature
    )
    results['generative'] = {
        'validity_rate': validity_rate,
        'valid_count': gen_stats['valid_count'],
        'total_count': gen_stats['total_generated'],
        'unique_count': gen_stats['unique_count'],
        'avg_length': gen_stats['avg_length'],
        'samples': all_samples,
        'valid_samples': valid_samples,
    }
    
    # Combined summary
    results['summary'] = {
        'val_loss': tf_loss,
        'val_acc': tf_acc,
        'real_validity': validity_rate,  # THIS IS WHAT MATTERS!
    }
    
    return results


def sample_and_validate(model: ActorModelTransformer, n_samples: int = 100) -> Tuple[List[str], float]:
    """
    Sample SMILES from the model and compute validity rate.
    
    Legacy function for backward compatibility.
    Use validate_with_generation() for more detailed stats.
    
    Returns:
        samples: List of generated SMILES
        validity: Fraction of valid SMILES (0-1)
    """
    all_samples, valid_samples, validity, _ = validate_with_generation(
        model, n_samples=n_samples, temperature=1.0
    )
    return all_samples, validity


def main():
    parser = argparse.ArgumentParser(
        description="Supervised pre-training for SMILES Transformer"
    )
    
    # Required arguments
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to SMILES file (one SMILES per line)"
    )
    parser.add_argument(
        "--prior", type=str, required=True,
        help="Path to pre-trained prior model (for vocabulary)"
    )
    
    # Model architecture (MUST match RL framework defaults for compatibility!)
    # RL defaults: layer_size=256, n_layers=6, n_heads=8, dropout=0.0
    parser.add_argument("--layer-size", type=int, default=256, help="Transformer hidden size (must match RL config)")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of transformer layers (must match RL config)")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads (must match RL config)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate (must match RL config, default 0.0)")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=5, help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument("--max-length", type=int, default=120, help="Maximum sequence length")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    
    # Early stopping
    parser.add_argument(
        "--early-stopping", action="store_true",
        help="Enable early stopping when validation loss stops improving"
    )
    parser.add_argument(
        "--patience", type=int, default=2,
        help="Early stopping patience: epochs to wait without improvement (default: 2)"
    )
    parser.add_argument(
        "--min-delta", type=float, default=0.01,
        help="Minimum improvement in val loss to reset patience (default: 0.01)"
    )
    
    # Output
    parser.add_argument(
        "--save-path", type=str, default="checkpoints/pretrained_transformer.pt",
        help="Path to save the best model"
    )
    parser.add_argument(
        "--save-every-epoch", action="store_true",
        help="Save checkpoint after every epoch (checkpoint_epoch_X.pt)"
    )
    parser.add_argument("--log-interval", type=int, default=50, help="Log every N batches")
    
    # Generative validation (addresses exposure bias)
    parser.add_argument(
        "--gen-samples", type=int, default=100,
        help="Number of molecules to generate for validity check each epoch (default: 100)"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature for generation (1.0=standard, 0.8=conservative)"
    )
    parser.add_argument(
        "--save-on-validity", action="store_true", default=True,
        help="Save best model based on REAL validity (not loss). Recommended!"
    )
    parser.add_argument(
        "--save-on-loss", action="store_true",
        help="Save best model based on validation loss (traditional, less reliable)"
    )
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    
    args = parser.parse_args()
    
    # Set random seed
    debug_print(f"[DEBUG] Setting random seed: {args.seed}")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device setup
    debug_print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
    debug_print(f"[DEBUG] no_cuda flag: {args.no_cuda}")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    debug_print(f"[DEBUG] Using device: {device}")
    
    if device.type == 'cuda':
        debug_print(f"[DEBUG] GPU: {torch.cuda.get_device_name(0)}")
        debug_print(f"[DEBUG] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create output directory
    save_dir = Path(args.save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    debug_print(f"[DEBUG] Output directory: {save_dir}")
    
    # ==================== Load Vocabulary ====================
    debug_print("\n" + "=" * 60)
    debug_print("STEP 1: Loading Vocabulary")
    debug_print("=" * 60)
    debug_print(f"[DEBUG] Prior path: {args.prior}")
    
    try:
        vocabulary, tokenizer, max_seq_len = load_vocab(args.prior)
        debug_print(f"[DEBUG] Vocabulary loaded successfully!")
        debug_print(f"  Vocabulary size: {len(vocabulary)}")
        debug_print(f"  Max sequence length: {max_seq_len}")
    except Exception as e:
        debug_print(f"[ERROR] Failed to load vocabulary: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Get padding token index (should be 0 = '$')
    pad_token_idx = vocabulary["$"]
    debug_print(f"  Padding token index: {pad_token_idx}")
    debug_print(f"  Start token index: {vocabulary['^']}")
    
    # ==================== Create Model ====================
    debug_print("\n" + "=" * 60)
    debug_print("STEP 2: Creating Model")
    debug_print("=" * 60)
    
    network_params = {
        'layer_size': args.layer_size,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'dropout': args.dropout,
    }
    debug_print(f"[DEBUG] Network params: {network_params}")
    
    try:
        model = create_model(
            vocabulary=vocabulary,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            network_params=network_params,
            device=device,
        )
        debug_print("[DEBUG] Model created successfully!")
    except Exception as e:
        debug_print(f"[ERROR] Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ==================== Load Data ====================
    debug_print("\n" + "=" * 60)
    debug_print("STEP 3: Loading Data")
    debug_print("=" * 60)
    debug_print(f"[DEBUG] Data path: {args.data}")
    
    if not os.path.exists(args.data):
        debug_print(f"[ERROR] Data file not found: {args.data}")
        sys.exit(1)
    
    smiles_list = load_smiles_from_file(args.data)
    debug_print(f"[DEBUG] Loaded {len(smiles_list)} SMILES from {args.data}")
    
    # Create dataset
    debug_print("[DEBUG] Creating SMILESDataset...")
    try:
        dataset = SMILESDataset(
            smiles_list,
            vocabulary,
            tokenizer,
            max_length=args.max_length,
        )
        debug_print(f"[DEBUG] Dataset created with {len(dataset)} samples")
    except Exception as e:
        debug_print(f"[ERROR] Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Train/val split
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    
    debug_print(f"[DEBUG] Splitting dataset: {n_train} train, {n_val} val")
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    debug_print(f"[DEBUG] Train: {n_train} samples, Val: {n_val} samples")
    
    # Create DataLoaders
    debug_print("[DEBUG] Creating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=SMILESDataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    debug_print(f"[DEBUG] Train loader: {len(train_loader)} batches")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=SMILESDataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    debug_print(f"[DEBUG] Val loader: {len(val_loader)} batches")
    
    # ==================== Setup Training ====================
    debug_print("\n" + "=" * 60)
    debug_print("STEP 4: Setting up Training")
    debug_print("=" * 60)
    
    # Loss function with padding masking
    # CRITICAL FIX: Use ignore_index=-100 (NOT 0!)
    # The dataset now pads targets with -100, so padding is ignored
    # But EOS (index 0) is now LEARNED because it's not ignored!
    # This was the bug causing 0% validity - EOS was never learned!
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    debug_print(f"[DEBUG] Loss: CrossEntropyLoss(ignore_index=-100) - EOS will now be learned!")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.transformer.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    debug_print(f"[DEBUG] Optimizer: AdamW(lr={args.lr}, weight_decay={args.weight_decay})")
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    debug_print(f"[DEBUG] Scheduler: CosineAnnealingLR(T_max={total_steps})")
    
    # ==================== Training Loop ====================
    debug_print("\n" + "=" * 60)
    debug_print(f"STEP 5: Starting Training for {args.epochs} Epochs")
    debug_print("=" * 60)
    
    # Initialize training history tracker
    history = TrainingHistory(save_dir=save_dir)
    debug_print(f"[DEBUG] Training history will be saved to: {save_dir}")
    
    # Initialize visualizer for plots and molecule grids
    visualizer = TrainingVisualizer(save_dir=save_dir)
    debug_print(f"[DEBUG] Visualizations will be saved to: {save_dir / 'visualizations'}")
    
    # Initialize early stopping if enabled
    early_stopper = None
    if args.early_stopping:
        early_stopper = EarlyStopping(
            patience=args.patience, 
            min_delta=args.min_delta, 
            verbose=True
        )
        debug_print(f"[DEBUG] Early stopping enabled: patience={args.patience}, min_delta={args.min_delta}")
    
    # Determine checkpoint saving strategy
    # IMPORTANT: Save based on VALIDITY by default (addresses exposure bias!)
    save_on_validity = args.save_on_validity and not args.save_on_loss
    if save_on_validity:
        debug_print("[DEBUG] Checkpoint strategy: Save best model based on REAL VALIDITY (recommended!)")
    else:
        debug_print("[DEBUG] Checkpoint strategy: Save best model based on validation loss")
    
    best_val_loss = float('inf')
    best_validity = 0.0
    best_epoch = 0
    stopped_early = False
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        print(f"\n{'='*60}", flush=True)
        print(f"EPOCH {epoch}/{args.epochs}", flush=True)
        print(f"{'='*60}", flush=True)
        
        # Train with CAUSAL MASKING (critical for autoregressive generation!)
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.log_interval,
            pad_idx=pad_token_idx
        )
        
        # Step scheduler (per epoch)
        scheduler.step()
        
        # ==================== COMPLETE VALIDATION ====================
        # This addresses EXPOSURE BIAS by doing BOTH:
        # 1. Teacher-forced validation (with causal mask - matches training)
        # 2. Generative validation (REAL performance - the true test!)
        
        # CRITICAL: Clear GPU memory before validation to avoid OOM during sampling
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            debug_print("[DEBUG] Cleared CUDA cache before validation")
        
        print("\n" + "-" * 40)
        print("VALIDATION (Teacher-Forced + Generative)")
        print("-" * 40)
        
        val_results = validate_complete(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            n_gen_samples=args.gen_samples,
            temperature=args.temperature,
            pad_idx=pad_token_idx,
        )
        
        # Extract metrics
        val_loss = val_results['teacher_forced']['loss']
        val_acc = val_results['teacher_forced']['accuracy']
        real_validity = val_results['generative']['validity_rate']
        valid_samples = val_results['generative']['valid_samples']
        all_samples = val_results['generative']['samples']
        unique_count = val_results['generative']['unique_count']
        
        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]
        
        # Print comprehensive summary
        print(f"\nEpoch {epoch} Results:", flush=True)
        print(f"  ┌─────────────────────────────────────────────────────┐")
        print(f"  │ TEACHER-FORCED (optimistic due to exposure bias):  │")
        print(f"  │   Loss: {val_loss:.4f}  |  Token Acc: {val_acc:.1f}%              │")
        print(f"  ├─────────────────────────────────────────────────────┤")
        print(f"  │ AUTOREGRESSIVE GENERATION (TRUE performance!):     │")
        print(f"  │   *** REAL Validity: {real_validity*100:.1f}% ***                    │")
        print(f"  │   Valid: {val_results['generative']['valid_count']}/{val_results['generative']['total_count']}  |  Unique: {unique_count}               │")
        print(f"  └─────────────────────────────────────────────────────┘")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"  LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        
        # Show sample generated molecules
        print(f"\n  Sample Generated SMILES (temp={args.temperature}):")
        for i, smi in enumerate(all_samples[:5]):
            # Check if valid
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smi)
                status = "✓ VALID" if mol else "✗ INVALID"
            except:
                status = "?"
            print(f"    {i+1}. [{status}] {smi[:60]}{'...' if len(smi)>60 else ''}")
        
        # Generate visualizations
        print("\n  Generating visualizations...")
        samples, vis_validity = visualizer.update(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            model=model,
            n_samples=64,
        )
        
        # Record metrics in history
        history.add_epoch(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            validity=real_validity,  # Use the REAL validity!
            lr=current_lr,
            epoch_time=epoch_time,
        )
        
        # ==================== CHECKPOINT SAVING ====================
        # ALWAYS save a checkpoint every epoch with descriptive name
        # This ensures we never lose progress due to OOM or other issues
        
        # 1. ALWAYS save epoch checkpoint (regardless of improvement)
        epoch_checkpoint = save_dir / f"checkpoint_epoch_{epoch}_loss{val_loss:.4f}_acc{val_acc:.1f}_val{real_validity*100:.0f}pct.pt"
        model.save(str(epoch_checkpoint))
        print(f"\n  📁 Epoch {epoch} checkpoint saved: {epoch_checkpoint.name}")
        
        # 2. Also save as "latest" for easy access
        latest_checkpoint = save_dir / "checkpoint_latest.pt"
        model.save(str(latest_checkpoint))
        
        # 3. Save as "best" if validity improved
        if save_on_validity:
            if real_validity > best_validity:
                improvement = real_validity - best_validity
                best_validity = real_validity
                best_epoch = epoch
                model.save(args.save_path)
                print(f"  ★★★ BEST MODEL SAVED! ★★★")
                print(f"  ★ REAL Validity improved: {(real_validity - improvement)*100:.1f}% → {real_validity*100:.1f}%")
                print(f"  ★ Saved to: {args.save_path}")
            else:
                print(f"  (Validity {real_validity*100:.1f}% did not improve over best {best_validity*100:.1f}% from epoch {best_epoch})")
        else:
            # Traditional: save based on loss
            if val_loss < best_val_loss:
                best_epoch = epoch
                model.save(args.save_path)
                print(f"  *** Best model saved (by loss): {args.save_path} (val_loss={val_loss:.4f}) ***")
        
        # Track best loss regardless (for logging)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        # Save history after each epoch (for crash recovery)
        history.save_csv()
        history.save_json()
        
        # Check early stopping (based on validity if save_on_validity, else loss)
        if early_stopper is not None:
            # Use validity for early stopping when saving on validity
            metric_for_stopping = -real_validity if save_on_validity else val_loss
            if early_stopper(metric_for_stopping, epoch):
                print(f"\n{'!'*60}")
                print(f"EARLY STOPPING triggered at epoch {epoch}")
                if save_on_validity:
                    print(f"Best model was at epoch {best_epoch} with validity={best_validity*100:.1f}%")
                else:
                    print(f"Best model was at epoch {early_stopper.best_epoch} with val_loss={early_stopper.best_loss:.4f}")
                print(f"{'!'*60}")
                stopped_early = True
                break
        
        print("-" * 60, flush=True)
    
    # ==================== Final Summary ====================
    
    # Save final history and generate plots
    csv_path = history.save_csv()
    json_path = history.save_json()
    plot_path = history.plot()
    
    # Generate final summary visualization
    visualizer.create_summary_plot()
    
    # Print summary
    history.print_summary()
    
    # Report early stopping status and best model info
    print("\n" + "=" * 60)
    print("BEST MODEL SUMMARY")
    print("=" * 60)
    if save_on_validity:
        print(f"★ Saved based on: REAL VALIDITY (addresses exposure bias)")
        print(f"★ Best epoch: {best_epoch}")
        print(f"★ Best validity: {best_validity * 100:.1f}%")
    else:
        print(f"Saved based on: Validation loss")
        print(f"Best epoch: {best_epoch}")
        print(f"Best loss: {best_val_loss:.4f}")
    
    if stopped_early:
        print(f"\n*** Training stopped early at epoch {len(history.epochs)} ***")
    
    # Final sampling with detailed stats
    print("\n" + "-" * 60)
    print("FINAL GENERATIVE VALIDATION (100 molecules)")
    print("-" * 60)
    all_samples, valid_samples, validity, stats = validate_with_generation(
        model, n_samples=100, temperature=args.temperature
    )
    print(f"  Temperature: {args.temperature}")
    print(f"  Total generated: {stats['total_generated']}")
    print(f"  Valid molecules: {stats['valid_count']} ({validity * 100:.1f}%)")
    print(f"  Invalid molecules: {stats['invalid_count']}")
    print(f"  Unique valid: {stats['unique_count']}")
    print(f"  Avg length: {stats['avg_length']:.1f}")
    
    # Warn if validity is low
    if validity < 0.5:
        print(f"\n⚠️  WARNING: Low validity ({validity*100:.1f}%)!")
        print(f"⚠️  Consider training longer or checking your data quality.")
    
    print(f"\nModel saved to: {args.save_path}")
    print(f"\nTraining artifacts saved to {save_dir}:")
    print(f"  - {csv_path.name} (CSV for plotting)")
    print(f"  - {json_path.name} (JSON with all metrics)")
    if plot_path:
        print(f"  - {plot_path.name} (training plots)")
    
    # List visualization files
    viz_dir = save_dir / "visualizations"
    print(f"\nVisualizations saved to {viz_dir}:")
    print(f"  - loss_curve.png (training & validation loss)")
    print(f"  - validity_curve.png (SMILES validity progress)")
    print(f"  - training_summary.png (combined summary)")
    print(f"  - samples.txt (generated SMILES log)")
    print(f"  - epoch_N_samples.png (molecule grid images per epoch)")
    
    print("\n" + "=" * 60)
    print("HOW TO USE WITH RL FRAMEWORK")
    print("=" * 60)
    print("\nUpdate your RL config JSON with these settings:")
    print(f'''
  "reinforcement_learning": {{
    "method": "smiles_rl.agent.a2c_transformer.A2CTransformer",
    "parameters": {{
      "agent": "{args.prior}",
      "prior": "{args.prior}",
      "specific_parameters": {{
        "transformer_weights": "{args.save_path}",
        ...
      }}
    }}
  }}
''')
    print("Then run RL with:")
    print("  python run_transformer.py --config config_transformer_pretrained.json")
    print("\nIMPORTANT: The pretrained model MUST use the same architecture as RL:")
    print(f"  - layer_size: {args.layer_size}")
    print(f"  - n_layers: {args.n_layers}")
    print(f"  - n_heads: {args.n_heads}")
    print(f"  - dropout: {args.dropout}")


if __name__ == "__main__":
    main()
