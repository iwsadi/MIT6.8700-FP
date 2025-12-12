import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import os
import random
import matplotlib.pyplot as plt
from rdkit import Chem

# Imports from existing files
from gcpn_model import GCPNPolicy
from create_trajectory import MoleculeTrajectory, ATOM_TYPES, BOND_TYPES

# Heuristic max valence for masking invalid bond targets during training
MAX_VALENCE = {
    1: 1,
    6: 4,
    7: 4,
    8: 2,
    9: 1,
    15: 5,
    16: 6,
    17: 1,
    35: 1,
    53: 1,
}

# Ensure we have a place to save models and figures
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("figures", exist_ok=True)

def save_plot(metrics, title, filename):
    # Robust plotting: metrics may contain tensors; convert to Python floats
    ys = []
    for m in metrics:
        if isinstance(m, torch.Tensor):
            ys.append(m.detach().float().cpu().item())
        else:
            ys.append(float(m))

    plt.figure()
    plt.plot(ys)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.grid(True)
    plt.savefig(os.path.join("figures", filename))
    plt.close()

def process_trajectory_step(step_data):
    """
    Converts a (Data, Action_Dict) pair into a Data object with ground truth labels.
    """
    data, action = step_data
    
    # Initialize labels
    y_stop = 0
    y_add_atom = -1
    y_edge_target = -1
    y_bond_type = -1  # index in BOND_TYPES (+1 for other)
    # store atomic numbers for valence masking
    atom_numbers = []
    
    mask_stop = True # Always train stop/go
    mask_add_atom = False
    mask_add_bond_type = False # NEW: For predicting bond type at Add Node step
    mask_edge_selection = False
    mask_bond_type = False
    
    if action['type'] == 'stop':
        y_stop = 1
        # No other losses
        
    elif action['type'] == 'add_atom':
        y_stop = 0
        mask_add_atom = True
        
        # Map atomic num to index
        atom_type = action['atom_type']
        if atom_type in ATOM_TYPES:
            y_add_atom = ATOM_TYPES.index(atom_type)
        else:
            y_add_atom = len(ATOM_TYPES) # Other
        atom_numbers.append(atom_type)

        # NEW: Handle bond_type for Add Atom
        # Check if 'bond_type' exists in action (added in create_trajectory.py)
        if 'bond_type' in action and action['bond_type'] is not None:
             mask_add_bond_type = True
             bond_type = action['bond_type']
             if bond_type in BOND_TYPES:
                 y_bond_type = BOND_TYPES.index(bond_type)
             else:
                 y_bond_type = len(BOND_TYPES)
            
    elif action['type'] == 'add_bond':
        y_stop = 0
        mask_edge_selection = True
        mask_bond_type = True
        
        y_edge_target = action['target_node_idx']
        # Map bond type to index
        bond_type = action.get('bond_type', Chem.rdchem.BondType.SINGLE)
        if bond_type in BOND_TYPES:
            y_bond_type = BOND_TYPES.index(bond_type)
        else:
            y_bond_type = len(BOND_TYPES)  # other/unknown
        
    # Attach to data
    data.y_stop = torch.tensor([y_stop], dtype=torch.float)
    data.y_add_atom = torch.tensor([y_add_atom], dtype=torch.long)
    data.y_edge_target = torch.tensor([y_edge_target], dtype=torch.long)
    data.y_bond_type = torch.tensor([y_bond_type], dtype=torch.long)
    
    data.mask_stop = torch.tensor([mask_stop], dtype=torch.bool)
    data.mask_add_atom = torch.tensor([mask_add_atom], dtype=torch.bool)
    data.mask_add_bond_type = torch.tensor([mask_add_bond_type], dtype=torch.bool)
    data.mask_edge_selection = torch.tensor([mask_edge_selection], dtype=torch.bool)
    data.mask_bond_type = torch.tensor([mask_bond_type], dtype=torch.bool)

    # Compute degrees and valid-to-add mask for valence supervision
    # Recover atomic numbers from node_features one-hot
    if data.x.size(0) > 0:
        x_np = data.x.numpy()
        atomic_nums = []
        for row in x_np:
            idx = int(np.argmax(row))
            if idx < len(ATOM_TYPES):
                atomic_nums.append(ATOM_TYPES[idx])
            else:
                atomic_nums.append(6)  # default C for unknown
    else:
        atomic_nums = []

    # Degrees from undirected edges
    degrees = [0] * len(atomic_nums)
    edge_pairs = set()
    for (u, v) in zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()):
        if u == v:
            continue
        key = tuple(sorted((u, v)))
        if key in edge_pairs:
            continue
        edge_pairs.add(key)
        # bond order from edge_attr if available
        bo = 1
        # find corresponding edge_attr row (first occurrence)
        try:
            idx = data.edge_index.t().tolist().index([u, v])
            if idx < data.edge_attr.size(0):
                one_hot = data.edge_attr[idx].tolist()
                bt_idx = int(np.argmax(one_hot))
                if bt_idx < len(BOND_TYPES):
                    bt = BOND_TYPES[bt_idx]
                    if bt == Chem.rdchem.BondType.DOUBLE:
                        bo = 2
                    elif bt == Chem.rdchem.BondType.TRIPLE:
                        bo = 3
                    elif bt == Chem.rdchem.BondType.AROMATIC:
                        bo = 2
        except Exception:
            pass
        degrees[u] += bo
        degrees[v] += bo

    valid_to_add = []
    for i, z in enumerate(atomic_nums):
        max_v = MAX_VALENCE.get(z, float('inf'))
        valid_to_add.append(degrees[i] < max_v)
    data.valid_to_add = torch.tensor(valid_to_add, dtype=torch.bool)
    
    # Ensure focus_node is a tensor, default to -1 if missing
    if hasattr(data, 'focus_node'):
        data.focus_node = torch.tensor([data.focus_node], dtype=torch.long)
    else:
        data.focus_node = torch.tensor([-1], dtype=torch.long)
        
    return data

class MoleculeDataset(Dataset):
    def __init__(self, smiles_list):
        self.smiles_list = smiles_list
        
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        # On-the-fly generation (Pure Lazy Loading)
        smiles = self.smiles_list[idx]
        try:
            # This runs on CPU workers in parallel
            traj_gen = MoleculeTrajectory(smiles)
            traj = traj_gen.generate_steps()
            processed_steps = [process_trajectory_step(step) for step in traj]
            return processed_steps # Returns a LIST of Data objects
        except Exception:
            return []

def flatten_collate_fn(batch):
    """
    Custom collate function to handle list of lists of Data objects.
    Flattens them into a single list for Batch.from_data_list.
    """
    flat_list = []
    for steps in batch:
        # steps is a list of Data objects
        if steps:
            flat_list.extend(steps)
    
    if not flat_list:
        # Return a dummy empty batch or handle it? 
        return None
        
    return Batch.from_data_list(flat_list)

def check_metrics(epoch, avg_loss, stop_acc, add_node_acc):
    """
    Sanity check for training metrics.
    """
    print(f"--- Metric Check Epoch {epoch+1} ---")
    
    # 1. Loss Check
    if epoch < 3:
        if avg_loss > 10.0:
            print(f"WARNING: Loss is unusually high ({avg_loss:.4f}). Check learning rate or gradients.")
    else:
        if avg_loss > 5.0:
             print(f"WARNING: Loss is not converging ({avg_loss:.4f}).")
             
    # 2. Stop Accuracy Check
    if stop_acc < 0.70:
        print(f"WARNING: Stop Accuracy is low ({stop_acc:.4f}). Model might be randomly terminating.")
        
    # 3. Add Node Accuracy Check
    if add_node_acc < 0.20:
         print(f"WARNING: Add Atom Accuracy is very low ({add_node_acc:.4f}). Check atom mapping.")
         
    # 4. Success check (Late stage)
    if epoch > 10 and stop_acc > 0.95 and add_node_acc > 0.80:
        print("INFO: Metrics look healthy and converged.")

def train():
    # 1. Load Data
    smiles_path = 'train_smiles.txt'
    if not os.path.exists(smiles_path):
        smiles_path = os.path.join('..', 'train_smiles.txt')
    
    if not os.path.exists(smiles_path):
         raise FileNotFoundError("Could not find train_smiles.txt in current or parent directory.")

    with open(smiles_path, 'r') as f:
        # Load all SMILES into memory (list of strings is cheap)
        smiles_list = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(smiles_list)} SMILES strings.")
    
    dataset = MoleculeDataset(smiles_list)
    
    # Optimization 1: Increase Batch Size
    BATCH_SIZE_MOLECULES = 32 
    
    # Optimization 2: Parallel Data Loading
    # On-the-fly generation requires CPU power. 
    # Use 4 workers to balance speed vs RAM usage (8 was causing crashes).
    NUM_WORKERS = 4
    
    from torch.utils.data import DataLoader as TorchDataLoader
    loader = TorchDataLoader(dataset, 
                           batch_size=BATCH_SIZE_MOLECULES, 
                           shuffle=True, 
                           collate_fn=flatten_collate_fn,
                           num_workers=NUM_WORKERS,
                           pin_memory=True) # pin_memory speeds up CPU->GPU transfer
    
    # 2. Model Setup
    input_dim = len(ATOM_TYPES) + 1
    hidden_dim = 64
    num_atom_types = len(ATOM_TYPES) + 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = GCPNPolicy(input_dim, hidden_dim, num_atom_types).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Add Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Losses
    criterion_stop = nn.BCEWithLogitsLoss()
    criterion_add_node = nn.CrossEntropyLoss()
    criterion_bond_type = nn.CrossEntropyLoss()
    
    # Metrics
    epoch_losses = []
    epoch_stop_acc = []
    epoch_add_node_acc = []
    
    # 3. Training Loop
    epochs = 20
    start_epoch = 0
    
    # Check for existing checkpoints to resume
    # We look for "gcpn_prior_epoch_X.pt"
    import glob
    checkpoints = glob.glob("checkpoints/gcpn_prior_epoch_*.pt")
    if checkpoints:
        # Find latest
        latest_ckpt = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(f"Resuming from checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        epoch_losses = checkpoint.get('epoch_losses', [])
        epoch_stop_acc = checkpoint.get('epoch_stop_acc', [])
        epoch_add_node_acc = checkpoint.get('epoch_add_node_acc', [])
        print(f"Resuming at Epoch {start_epoch + 1}")
    
    model.train()
    
    # Optimization 3: Mixed Precision (AMP)
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        total_steps = 0
        
        # Tracking accuracy within epoch
        stop_correct = 0
        stop_total = 0
        add_node_correct = 0
        add_node_total = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            if batch is None:
                continue
            
            # Move batch to device
            batch = batch.to(device)
                
            optimizer.zero_grad()
            
            # Use AMP Autocast
            with torch.cuda.amp.autocast():
            
                # Prepare new_node_indices logic
                new_node_indices = None
                focus_node_indices = None # NEW: For Add Bond prediction
    
                if hasattr(batch, 'focus_node'):
                    ptr = batch.ptr[:-1] # Start indices
                    focus_nodes = batch.focus_node # [batch_size]
                    
                    # Check consistency
                    if len(focus_nodes) != len(ptr):
                        pass
    
                    global_focus_nodes = ptr + focus_nodes
                    # Clamp negatives to 0 to avoid index errors, will be masked later
                    global_focus_nodes = torch.where(focus_nodes >= 0, global_focus_nodes, torch.zeros_like(global_focus_nodes))
                    
                    # Assign to both (conceptually same node, different context usage)
                    new_node_indices = global_focus_nodes
                    focus_node_indices = global_focus_nodes # Used for Add Atom bond prediction
                
                outputs = model(batch, new_node_indices=new_node_indices, focus_node_indices=focus_node_indices)
                
                # --- Calculate Loss ---
                loss = torch.tensor(0.0, device=device) # Initialize as CUDA tensor
                
                # 1. Stop Loss
                stop_logits = outputs['stop_logits'].squeeze(-1) # [batch_size]
                # Ensure target shape matches logits
                stop_targets = batch.y_stop # [batch_size]
                
                stop_loss = criterion_stop(stop_logits, stop_targets)
                loss += stop_loss
                
                # Accuracy Stop
                preds_stop = (torch.sigmoid(stop_logits) > 0.5).float()
                stop_correct += (preds_stop == stop_targets).sum().item()
                stop_total += len(stop_targets)
                
                # 2. Add Node Loss
                if batch.mask_add_atom.any():
                    pred = outputs['add_node_logits'][batch.mask_add_atom]
                    target = batch.y_add_atom[batch.mask_add_atom]
                    add_loss = criterion_add_node(pred, target)
                    loss += add_loss
                    
                    # Accuracy Add Node
                    preds_add = torch.argmax(pred, dim=1)
                    add_node_correct += (preds_add == target).sum().item()
                    add_node_total += len(target)
    
                    # NEW: Add Bond Type Loss (Connection to Parent)
                    # Only if mask_add_bond_type is present and true
                    if hasattr(batch, 'mask_add_bond_type') and batch.mask_add_bond_type.any():
                         bond_mask = batch.mask_add_bond_type
                         bond_pred = outputs['add_bond_logits'][bond_mask]
                         bond_target = batch.y_bond_type[bond_mask]
                         
                         add_bond_loss = criterion_bond_type(bond_pred, bond_target)
                         loss += add_bond_loss
                
                # 3. Edge Selection Loss
                if batch.mask_edge_selection.any():
                    mask = batch.mask_edge_selection
                    from torch_geometric.utils import softmax as pyg_softmax
                    
                    scores = outputs['edge_selection_logits'].squeeze(-1) # [total_nodes]
                # Valence mask: set invalid nodes to large negative to avoid assigning prob
                if hasattr(batch, 'valid_to_add'):
                    node_mask = batch.valid_to_add
                    # Use -1e4 instead of -1e9 for float16 compatibility (AMP)
                    scores = scores.masked_fill(~node_mask, -1e4)

                    # Log Softmax per graph
                    probs = pyg_softmax(scores, batch.batch)
                    log_probs = torch.log(probs + 1e-10)
                    
                    # Select target log probs
                    ptr = batch.ptr[:-1]
                    global_targets = ptr + batch.y_edge_target
                    
                    valid_global_targets = global_targets[mask]
                    # If valence mask exists, drop targets that are invalid
                    if hasattr(batch, 'valid_to_add'):
                        valid_target_mask = batch.valid_to_add[valid_global_targets]
                        valid_global_targets = valid_global_targets[valid_target_mask]
                    if valid_global_targets.numel() == 0:
                        edge_loss = torch.tensor(0.0, device=scores.device)
                        loss += edge_loss
                    else:
                        selected_log_probs = log_probs[valid_global_targets]
                        edge_loss = -selected_log_probs.mean()
                        loss += edge_loss
                    
                    # Bond type loss (use bond_type_logits gathered at target nodes)
                    bond_logits = outputs.get('bond_type_logits', None)
                    if bond_logits is not None and valid_global_targets.numel() > 0:
                        bond_targets = batch.y_bond_type[mask]
                        if hasattr(batch, 'valid_to_add'):
                            bond_targets = bond_targets[valid_target_mask]
                        target_logits = bond_logits[valid_global_targets]
                        bond_loss = criterion_bond_type(target_logits, bond_targets)
                        loss += bond_loss
            
            # Replaced loss.backward() and optimizer.step() with AMP scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_steps += 1
            
            pbar.set_postfix({'loss': total_loss / total_steps})

        # End of Epoch Metrics
        avg_loss = total_loss / total_steps
        epoch_losses.append(avg_loss)
        
        avg_stop_acc = stop_correct / stop_total if stop_total > 0 else 0
        epoch_stop_acc.append(avg_stop_acc)
        
        avg_add_node_acc = add_node_correct / add_node_total if add_node_total > 0 else 0
        epoch_add_node_acc.append(avg_add_node_acc)
        
        print(f"Epoch {epoch+1} Stats: Loss={avg_loss:.4f}, Stop Acc={avg_stop_acc:.4f}, Add Atom Acc={avg_add_node_acc:.4f}")
        
        # Run Sanity Check
        check_metrics(epoch, avg_loss, avg_stop_acc, avg_add_node_acc)

        # Step Scheduler
        scheduler.step(avg_loss)

        # Save Checkpoint every epoch
        checkpoint_path = f"checkpoints/gcpn_prior_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch_losses': epoch_losses,
            'epoch_stop_acc': epoch_stop_acc,
            'epoch_add_node_acc': epoch_add_node_acc
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Also update the main model file
        torch.save(model.state_dict(), "gcpn_prior.pt")

    # Save Model (Final)
    torch.save(model.state_dict(), "gcpn_prior.pt")
    print("Model saved to gcpn_prior.pt")
    
    # Save Figures
    save_plot(epoch_losses, "Training Loss", "loss.png")
    save_plot(epoch_stop_acc, "Stop Prediction Accuracy", "stop_accuracy.png")
    save_plot(epoch_add_node_acc, "Add Atom Prediction Accuracy", "atom_accuracy.png")
    print("Figures saved to GCPN/figures/")

if __name__ == "__main__":
    train()
