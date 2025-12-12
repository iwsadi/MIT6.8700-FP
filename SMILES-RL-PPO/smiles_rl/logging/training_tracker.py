"""
Training Tracker for PPO-based SMILES generation.
Tracks metrics, generates plots, and saves models for each training run.
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server/headless environments
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plotting disabled.")


class TrainingTracker:
    """
    Tracks training metrics and generates visualizations for PPO training runs.
    
    Creates a unique folder for each training run containing:
    - training.log: Copy of the training log
    - metrics.json: All tracked metrics in JSON format
    - scores_plot.png: Component scores and Reward vs steps (dynamically shows available components)
    - entropy_plot.png: Policy entropy vs steps
    - loss_plot.png: Critic and Actor loss vs steps
    - sequence_length_plot.png: Average sequence length vs steps
    - final_model.ckpt: The final trained model
    """
    
    def __init__(self, base_log_dir: str = "logs", run_name: Optional[str] = None, plot_frequency: int = 0):
        """
        Initialize the training tracker.
        
        Args:
            base_log_dir: Base directory for all training logs
            run_name: Optional custom name for this run. If None, uses timestamp.
            plot_frequency: Generate plots every N steps. 0 = only at end. (e.g., 1000 = every 1000 steps)
        """
        self.base_log_dir = base_log_dir
        self.plot_frequency = plot_frequency
        
        # Create unique run folder with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name:
            self.run_name = f"{timestamp}_{run_name}"
        else:
            self.run_name = timestamp
            
        self.run_dir = os.path.join(base_log_dir, "runs", self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize metric storage
        self.metrics: Dict[str, List] = {
            "step": [],
            "total_score": [],
            "classification_score": [],
            "qed_score": [],
            "validity": [],
            "actor_loss": [],
            "critic_loss": [],
            "policy_entropy": [],
            "kl_divergence": [],
            "avg_sequence_length": [],
            "kl_coeff": [],  # Track adaptive KL coefficient
            "top5_mean": [],  # Top-5 average score (最高5个分子的平均分)
            "max_score": [],  # Historical maximum score
        }
        
        # Store component names for flexible tracking
        self.component_names: List[str] = []
        
        if plot_frequency > 0:
            print(f"TrainingTracker initialized. Run directory: {self.run_dir}")
            print(f"  -> Plots will be saved every {plot_frequency} steps")
        else:
            print(f"TrainingTracker initialized. Run directory: {self.run_dir}")
    
    def log_step(
        self,
        step: int,
        total_score: float,
        component_scores: Dict[str, float],
        validity: float,
        actor_loss: float,
        critic_loss: float,
        policy_entropy: float,
        kl_divergence: float,
        avg_sequence_length: float,
        kl_coeff: float = 0.0,
        top5_mean: float = 0.0,
        max_score: float = 0.0,
    ):
        """
        Log metrics for a single training step.
        
        Args:
            step: Current training step
            total_score: Mean reward across batch (after diversity filter + invalid penalty)
            component_scores: Dict of component name -> mean score (e.g., {"classification": 0.1})
            validity: Fraction of valid SMILES (0-100)
            actor_loss: Actor (policy) loss
            critic_loss: Critic (value) loss
            policy_entropy: Policy entropy
            kl_divergence: KL divergence from prior
            avg_sequence_length: Average SMILES sequence length
            kl_coeff: Current adaptive KL coefficient
            top5_mean: Top-5 average score (最高5个分子的平均分)
            max_score: Historical maximum score across all steps
        """
        self.metrics["step"].append(step)
        self.metrics["total_score"].append(total_score)
        self.metrics["validity"].append(validity)
        self.metrics["actor_loss"].append(actor_loss)
        self.metrics["critic_loss"].append(critic_loss)
        self.metrics["policy_entropy"].append(policy_entropy)
        self.metrics["kl_divergence"].append(kl_divergence)
        self.metrics["avg_sequence_length"].append(avg_sequence_length)
        self.metrics["kl_coeff"].append(kl_coeff)
        self.metrics["top5_mean"].append(top5_mean)
        self.metrics["max_score"].append(max_score)
        
        # Handle component scores dynamically
        for name, score in component_scores.items():
            key = f"component_{name}"
            if key not in self.metrics:
                self.metrics[key] = []
                self.component_names.append(name)
            self.metrics[key].append(score)
        
        # Also store common component names for backward compatibility
        if "classification" in component_scores:
            self.metrics["classification_score"].append(component_scores["classification"])
        if "QED" in component_scores:
            self.metrics["qed_score"].append(component_scores["QED"])
        
        # Generate plots periodically if plot_frequency is set
        if self.plot_frequency > 0 and step > 0 and step % self.plot_frequency == 0:
            print(f"\n Generating interim plots at step {step}...")
            self.generate_plots()
            self.save_metrics()  # Also save metrics periodically
    
    def save_metrics(self):
        """Save all metrics to a JSON file."""
        # Ensure run directory exists (it might have been deleted or not created properly)
        os.makedirs(self.run_dir, exist_ok=True)
        
        metrics_path = os.path.join(self.run_dir, "metrics.json")
        
        # Convert numpy types to Python types for JSON serialization
        serializable_metrics = {}
        for key, values in self.metrics.items():
            serializable_metrics[key] = [
                float(v) if isinstance(v, (np.floating, np.integer)) else v 
                for v in values
            ]
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        print(f"Metrics saved to {metrics_path}")
    
    def copy_training_log(self, source_log_path: str):
        """Copy the training log to the run directory."""
        if os.path.exists(source_log_path):
            dest_path = os.path.join(self.run_dir, "training.log")
            shutil.copy2(source_log_path, dest_path)
            print(f"Training log copied to {dest_path}")
    
    def save_model(self, agent, filename: str = "final_model.ckpt"):
        """Save the trained model."""
        model_path = os.path.join(self.run_dir, filename)
        agent.save_to_file(model_path)
        print(f"Model saved to {model_path}")
    
    def save_config(self, config: dict):
        """Save the training configuration."""
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print(f"Config saved to {config_path}")
    
    def generate_plots(self):
        """Generate all training plots."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available. Skipping plot generation.")
            return
        
        if len(self.metrics["step"]) == 0:
            print("No data to plot.")
            return
        
        steps = self.metrics["step"]
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'ggplot')
        
        # 1. Scores Plot (all available components + Reward)
        self._plot_scores(steps)
        
        # 2. Policy Entropy Plot
        self._plot_entropy(steps)
        
        # 3. Loss Plot (Actor and Critic)
        self._plot_losses(steps)
        
        # 4. Sequence Length Plot
        self._plot_sequence_length(steps)
        
        # 5. Validity Plot
        self._plot_validity(steps)
        
        # 6. KL Divergence Plot
        self._plot_kl_divergence(steps)
        
        # 7. Combined Summary Plot
        self._plot_summary(steps)
        
        print(f"All plots saved to {self.run_dir}")
    
    def _plot_scores(self, steps):
        """Plot all component scores and Reward vs steps (dynamically handles available components)."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(steps, self.metrics["total_score"], 'b-', linewidth=2, label='Mean Score', alpha=0.8)
        
        # Plot Top-5 mean score (最高5个分子的平均分)
        if self.metrics["top5_mean"]:
            ax.plot(steps, self.metrics["top5_mean"], 'r-', linewidth=2, label='Top-5 Mean', alpha=0.8)
        
        if self.metrics["qed_score"]:
            ax.plot(steps, self.metrics["qed_score"], 'c-', linewidth=1.5, label='QED', alpha=0.7)
        
        if self.metrics["classification_score"]:
            ax.plot(steps, self.metrics["classification_score"], 'm-', linewidth=1.5, label='Classification (Activity)', alpha=0.7)
        
        # Plot any other components
        for name in self.component_names:
            if name not in ["classification", "QED"]:
                key = f"component_{name}"
                if self.metrics[key]:
                    ax.plot(steps, self.metrics[key], '--', linewidth=1, label=name, alpha=0.6)
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Scores vs Training Steps', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at y=0 for reference
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, "scores_plot.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_entropy(self, steps):
        """Plot Policy Entropy vs steps."""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(steps, self.metrics["policy_entropy"], 'purple', linewidth=2, alpha=0.8)
        ax.fill_between(steps, self.metrics["policy_entropy"], alpha=0.2, color='purple')
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Policy Entropy', fontsize=12)
        ax.set_title('Policy Entropy vs Training Steps', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, "entropy_plot.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_losses(self, steps):
        """Plot Actor and Critic losses vs steps."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Actor Loss
        ax1.plot(steps, self.metrics["actor_loss"], 'b-', linewidth=1.5, alpha=0.8)
        ax1.set_ylabel('Actor Loss', fontsize=12)
        ax1.set_title('Actor Loss vs Training Steps', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Use log scale if values vary widely
        actor_range = max(self.metrics["actor_loss"]) - min(self.metrics["actor_loss"])
        if actor_range > 1000:
            ax1.set_yscale('symlog')
        
        # Critic Loss
        ax2.plot(steps, self.metrics["critic_loss"], 'r-', linewidth=1.5, alpha=0.8)
        ax2.set_xlabel('Training Step', fontsize=12)
        ax2.set_ylabel('Critic Loss', fontsize=12)
        ax2.set_title('Critic Loss vs Training Steps', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, "loss_plot.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_sequence_length(self, steps):
        """Plot Average Sequence Length vs steps."""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(steps, self.metrics["avg_sequence_length"], 'orange', linewidth=2, alpha=0.8)
        ax.fill_between(steps, self.metrics["avg_sequence_length"], alpha=0.2, color='orange')
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Average Sequence Length', fontsize=12)
        ax.set_title('Average SMILES Length vs Training Steps', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add reference line for typical SMILES length
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Typical length (~50)')
        ax.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, "sequence_length_plot.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_validity(self, steps):
        """Plot Fraction of Valid SMILES vs steps."""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(steps, self.metrics["validity"], 'green', linewidth=2, alpha=0.8)
        ax.fill_between(steps, self.metrics["validity"], alpha=0.2, color='green')
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Valid SMILES (%)', fontsize=12)
        ax.set_title('Fraction of Valid SMILES vs Training Steps', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3)
        
        # Add reference lines
        ax.axhline(y=100, color='blue', linestyle='--', alpha=0.3, label='100% valid')
        ax.axhline(y=80, color='red', linestyle='--', alpha=0.3, label='Reset threshold (80%)')
        ax.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, "validity_plot.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_kl_divergence(self, steps):
        """Plot KL Divergence vs steps."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # KL Divergence
        ax1.plot(steps, self.metrics["kl_divergence"], 'teal', linewidth=2, alpha=0.8)
        ax1.set_ylabel('KL Divergence', fontsize=12)
        ax1.set_title('KL Divergence from Prior vs Training Steps', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.01, color='orange', linestyle='--', alpha=0.5, label='Target KL')
        ax1.legend(loc='best')
        
        # KL Coefficient (if adaptive)
        if any(k != 0 for k in self.metrics["kl_coeff"]):
            ax2.plot(steps, self.metrics["kl_coeff"], 'brown', linewidth=2, alpha=0.8)
            ax2.set_ylabel('KL Coefficient', fontsize=12)
            ax2.set_title('Adaptive KL Coefficient vs Training Steps', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'KL Coefficient not tracked (static)', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        
        ax2.set_xlabel('Training Step', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, "kl_divergence_plot.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_summary(self, steps):
        """Generate a combined summary plot with all key metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Scores
        ax = axes[0, 0]
        ax.plot(steps, self.metrics["total_score"], 'b-', linewidth=2, label='Mean', alpha=0.8)
        if self.metrics["top5_mean"]:
            ax.plot(steps, self.metrics["top5_mean"], 'r-', linewidth=2, label='Top-5', alpha=0.8)
        ax.set_xlabel('Step')
        ax.set_ylabel('Score')
        ax.set_title('Scores (Mean/Top-5)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 2. Validity
        ax = axes[0, 1]
        ax.plot(steps, self.metrics["validity"], 'green', linewidth=2, alpha=0.8)
        ax.fill_between(steps, self.metrics["validity"], alpha=0.2, color='green')
        ax.axhline(y=80, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Valid SMILES (%)')
        ax.set_title('Validity')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3)
        
        # 3. Actor Loss
        ax = axes[0, 2]
        ax.plot(steps, self.metrics["actor_loss"], 'b-', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Step')
        ax.set_ylabel('Actor Loss')
        ax.set_title('Actor Loss')
        if max(self.metrics["actor_loss"]) - min(self.metrics["actor_loss"]) > 1000:
            ax.set_yscale('symlog')
        ax.grid(True, alpha=0.3)
        
        # 4. Critic Loss
        ax = axes[1, 0]
        ax.plot(steps, self.metrics["critic_loss"], 'r-', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Step')
        ax.set_ylabel('Critic Loss')
        ax.set_title('Critic Loss')
        ax.grid(True, alpha=0.3)
        
        # 5. Entropy
        ax = axes[1, 1]
        ax.plot(steps, self.metrics["policy_entropy"], 'purple', linewidth=2, alpha=0.8)
        ax.set_xlabel('Step')
        ax.set_ylabel('Policy Entropy')
        ax.set_title('Policy Entropy')
        ax.grid(True, alpha=0.3)
        
        # 6. Sequence Length
        ax = axes[1, 2]
        ax.plot(steps, self.metrics["avg_sequence_length"], 'orange', linewidth=2, alpha=0.8)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Avg Length')
        ax.set_title('Sequence Length')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Training Summary - {self.run_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, "summary_plot.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    def finalize(self, agent=None, source_log_path: str = None, config: dict = None):
        """
        Finalize the training run: save metrics, generate plots, copy log, save model.
        
        Args:
            agent: The trained agent (optional, for saving the model)
            source_log_path: Path to the training log file (optional, for copying)
            config: Training configuration dict (optional, for saving)
        """
        print(f"\n{'='*60}")
        print(f"Finalizing training run: {self.run_name}")
        print(f"{'='*60}")
        
        # Save metrics
        self.save_metrics()
        
        # Generate plots
        self.generate_plots()
        
        # Copy training log
        if source_log_path:
            self.copy_training_log(source_log_path)
        
        # Save config
        if config:
            self.save_config(config)
        
        # Save model
        if agent:
            self.save_model(agent)
        
        print(f"\nTraining artifacts saved to: {self.run_dir}")
        print(f"{'='*60}\n")
        
        return self.run_dir
