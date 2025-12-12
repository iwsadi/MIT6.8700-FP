# MIT6.8700-FP

## Graph Convolutional Policy Network (GCPN)

A PyTorch implementation of a graph-generative policy network for drug-like molecule creation. This model treats molecule generation as a sequential decision process on a graph $G_t$, enforcing chemical validity via valence constraints.

### 🧠 Methodology

### 1. Trajectory Generation
We convert SMILES into supervised training trajectories $\{(G_t, f_t, a_t)\}$ using **Randomized Breadth-First Search (BFS)**:
* **Initialization:** Select random start atom ($G_0$).
* **Expansion:** Maintain a queue of "focus nodes" ($f_t$).
* **Actions:** `add_atom` (expand to neighbor), `add_bond` (ring closure), or `stop`.

### 2. Model Architecture
**Encoder:** An $L$-layer GNN (GCN/GAT backbone) processes node features $\mathbf{X}$ into node embeddings $\mathbf{H}$ and a pooled graph embedding $\mathbf{g}$.

**Policy Heads:**
The action space is decomposed into 5 MLP heads:
1.  **Stop:** $\ell_{\text{stop}}(\mathbf{g})$ (Binary probability)
2.  **Atom Type:** $\ell_{\text{atom}}(\mathbf{g})$ (Element prediction)
3.  **Bond Order:** $\ell_{\text{addbond}}(\mathbf{h}_f)$ (Bond type to parent)
4.  **Ring Closure:** $\ell_{\text{edge}}$ & $\ell_{\text{bond}}$ (Score existing nodes for connection)

### 3. Constraints & Optimization
* **Valence Masking:** Logits for chemically invalid connections are masked with $-\infty$ based on current node degree $\deg(v)$ and $\text{MaxValence}(z)$.
* **Objective:** We optimize a multi-task supervised loss:

```math
\mathcal{L} = \mathcal{L}_{\text{stop}} + \mathcal{L}_{\text{atom}} + \mathcal{L}_{\text{addbond}} + \mathcal{L}_{\text{edge}} + \mathcal{L}_{\text{bond}}
```

### 📐 Pipeline Overview

```mermaid
graph LR
    A[SMILES] -->|RDKit| B[Molecule]
    B -->|Rand BFS| C(Trajectory G_t, f_t)
    C -->|GNN Encoder| D[Embeddings H, g]
    D -->|Policy Heads| E[Logits: Stop, Atom, Bond, Edge]
```
## Installation
Same as the SMILES-RL setup. Use the environment within the GCPN folder:
```python
cd GCPN
conda env create -f env.yml
conda activate gcpn_env
```

## Usage

### Training GCPN
If you want to retrain the model, run the following command:
```python
python GCPN/pretrain.py
```
The file gcpn_prior.pt should be generated. The checkpoints are saved in GCPN/checkpoints in case the training fails. 

### Generation
```python
python GCPN/generate_gcpn.py
```

### Training Agent





