# SMILES-RL-PPO: Transformer-based Molecular Generation with PPO

A reinforcement learning framework for SMILES molecular generation using Proximal Policy Optimization (PPO) and Transformer models (MolGPT).

## Project Structure

```
SMILES-RL-PPO/
├── run_with_mock.py                # Main entry point
├── ppo_config_transformer.json     # PPO training configuration
├── pre_trained_models/
│   └── ChEMBL/
│       └── molgpt.prior            # Pre-trained MolGPT-Zinc15 model
├── predictive_models/
│   ├── create_DRD2_data_and_models.py  # Script to create DRD2 model
│   └── DRD2.csv                    # DRD2 activity data
├── logs/
│   ├── results/                    # Intermediate results
│   └── runs/                       # Training run outputs
└── smiles_rl/                      # Core source code
    ├── agent/                      # PPO reinforcement learning agent
    ├── model/                      # Transformer model (MolGPT)
    ├── diversity_filter/           # Diversity filters
    ├── logging/                    # Training loggers
    ├── replay_buffer/              # Experience replay
    └── scoring/                    # Scoring functions
```

## Installation

### Prerequisites

Create a conda environment with the required dependencies:

```bash
conda activate smiles_rl_test
pip install transformers accelerate
```

### Required Packages

- PyTorch (with CUDA support)
- RDKit
- reinvent-scoring
- reinvent-chemistry
- transformers
- accelerate

## Setup

### Generate DRD2 Predictive Model

Before running training, generate the DRD2 predictive model:

```bash
cd predictive_models
python create_DRD2_data_and_models.py
```

This will create `RF_DRD2_ecfp4c.pkl` in the `predictive_models/DRD2/` directory.

## Usage
 
Run PPO training with the configuration file:

```bash
python run_with_mock.py --config ppo_config_transformer.json
```

## Configuration

The `ppo_config_transformer.json` contains the following main settings:

### Reinforcement Learning Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 128 | Number of molecules generated per batch |
| `learning_rate` | 5e-6 | Learning rate for optimization |
| `n_steps` | 1000 | Total training steps |
| `clip` | 0.2 | PPO clipping parameter |
| `kl_coeff` | 2 | KL divergence penalty coefficient |
| `reward_scale` | 10.0 | Reward scaling factor |
| `entropy_coeff` | 0.01 | Entropy bonus coefficient |

### Diversity Filter
- Uses `IdenticalMurckoScaffold` to filter duplicate scaffolds
- `bucket_size`: 100 (max molecules per scaffold)
- `minscore`: 0.05 (minimum score threshold)

### Scoring Function
- Uses DRD2 activity prediction model (Random Forest classifier)
- Descriptor: ECFP4 counts (radius=2, size=2048)

### Replay Buffer
- Uses `TopHistory` to store highest-scoring molecules
- `k`: 32 (top molecules to keep)
- `memory_size`: 1000

## Training Output

Training results are saved in `logs/runs/<timestamp>_<job_id>/`:

| File | Description |
|------|-------------|
| `scores_plot.png` | Score progression over training |
| `summary_plot.png` | Training summary visualization |
| `final_model.ckpt` | Final model checkpoint |
| `metrics.json` | Training metrics |
| `training.log` | Detailed training log |
| `validity_plot.png` | SMILES validity over training |
| `kl_divergence_plot.png` | KL divergence monitoring |
| `entropy_plot.png` | Policy entropy over training |
| `loss_plot.png` | Actor and critic loss curves |

## Model Architecture

This framework uses **MolGPT-Zinc15** as the base transformer model for molecular generation:
- Pre-trained on ZINC15 dataset
- Fine-tuned using PPO with DRD2 activity as the reward signal
- Supports adaptive KL penalty to prevent catastrophic forgetting
