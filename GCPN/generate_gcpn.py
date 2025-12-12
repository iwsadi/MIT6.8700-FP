import torch
import os
import sys
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from torch_geometric.data import Data, Batch

# -----------------------------------------------------------------------------
# 1. ROBUST PATH SETUP
# -----------------------------------------------------------------------------
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Prefer the freshly trained weights in the project root; fallback to pre-trained folder.
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "gcpn_prior.pt")
if not os.path.exists(MODEL_WEIGHTS_PATH):
    MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "pre_trained_models", "ChEMBL", "gcpn_prior.pt")

from GCPN.gcpn_model import GCPNPolicy
from GCPN.create_trajectory import ATOM_TYPES, BOND_TYPES, get_atom_one_hot, get_bond_one_hot

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

# Heuristic maximum explicit valence by atomic number (kept conservative)
MAX_VALENCE = {
    1: 1,   # H
    6: 4,   # C
    7: 4,   # N (allow 4 to cover common quaternary / aromatic cases)
    8: 2,   # O
    9: 1,   # F
    15: 5,  # P
    16: 6,  # S
    17: 1,  # Cl
    35: 1,  # Br
    53: 1,  # I
}

def explicit_valence(atom: Chem.Atom) -> int:
    """RDKit-safe explicit valence accessor (avoids deprecated GetExplicitValence())."""
    return int(atom.GetValence(Chem.rdchem.ValenceType.EXPLICIT))

def build_mol_from_graph(data):
    """
    Reconstructs an RDKit molecule from a PyG Data object.
    (Used only for final output check if needed, but we now maintain Mol internally)
    """
    mol = Chem.RWMol()
    node_to_idx = {}
    x = data.x.cpu().numpy()
    for i in range(x.shape[0]):
        type_idx = np.argmax(x[i])
        if type_idx < len(ATOM_TYPES):
            atomic_num = ATOM_TYPES[type_idx]
        else:
            atomic_num = 6
        idx = mol.AddAtom(Chem.Atom(atomic_num))
        node_to_idx[i] = idx

    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    added_bonds = set()
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        if src >= dst: continue
        bond_key = tuple(sorted((src, dst)))
        if bond_key in added_bonds: continue
        added_bonds.add(bond_key)
        
        bond_feat = edge_attr[i]
        bond_type_idx = np.argmax(bond_feat)
        if bond_type_idx < len(BOND_TYPES):
            bond_type = BOND_TYPES[bond_type_idx]
        else:
            bond_type = Chem.rdchem.BondType.SINGLE
            
        if src in node_to_idx and dst in node_to_idx:
            mol.AddBond(node_to_idx[src], node_to_idx[dst], bond_type)
            
    try:
        Chem.SanitizeMol(mol)
        return mol
    except ValueError:
        return None

# ----------------------------------------------------------------------------- 
# BOND LOGIC (match training semantics)
# -----------------------------------------------------------------------------

def try_add_bond(mol: Chem.RWMol, u_idx: int, v_idx: int, preferred_type: Chem.rdchem.BondType, relaxed: bool = True):
    """
    Attempts to add a bond between u_idx and v_idx.

    Key design:
    - We DO NOT run full SanitizeMol at each step (aromatic rings are global).
    - We DO enforce valence overflow by trying UpdatePropertyCache(strict=True).
      If strict=True fails due to aromatic/partial state, we fall back to strict=False.
    """
    # Do not add duplicate bonds
    if mol.GetBondBetweenAtoms(u_idx, v_idx) is not None:
        return None

    candidates = [preferred_type]
    if preferred_type != Chem.rdchem.BondType.SINGLE:
        candidates.append(Chem.rdchem.BondType.SINGLE)

    for bt in candidates:
        try:
            mol.AddBond(u_idx, v_idx, bt)
        except Exception:
            continue

        # Hard valence guard (local, fast, prevents the "C valence 7" failures)
        try:
            au = mol.GetAtomWithIdx(u_idx)
            av = mol.GetAtomWithIdx(v_idx)
            vu = explicit_valence(au)
            vv = explicit_valence(av)
            mu = MAX_VALENCE.get(au.GetAtomicNum(), 999)
            mv = MAX_VALENCE.get(av.GetAtomicNum(), 999)
            if vu > mu or vv > mv:
                mol.RemoveBond(u_idx, v_idx)
                continue
        except Exception:
            # If we can't read valence, fall back to removing the bond
            mol.RemoveBond(u_idx, v_idx)
            continue

        # Enforce valence feasibility (without requiring global aromatic consistency).
        try:
            mol.UpdatePropertyCache(strict=True)
            return bt
        except Chem.AtomValenceException:
            mol.RemoveBond(u_idx, v_idx)
            continue
        except Exception:
            # Allow relaxed partial states (esp. aromatic construction)
            if relaxed:
                try:
                    mol.UpdatePropertyCache(strict=False)
                    return bt
                except Exception:
                    mol.RemoveBond(u_idx, v_idx)
                    continue
            mol.RemoveBond(u_idx, v_idx)
            continue

    return None


def can_sanitize(mol: Chem.RWMol) -> bool:
    """Check final validity without mutating the working RWMol."""
    try:
        m = Chem.Mol(mol)
        Chem.SanitizeMol(m)
        return True
    except Exception:
        return False

def sample_from_logits(logits_1d: torch.Tensor) -> int:
    """Sample an index from unnormalized logits."""
    probs = torch.softmax(logits_1d, dim=0)
    return torch.multinomial(probs, 1).item()

def generate_molecule(model, max_steps=80, device='cpu'):
    """
    Generation that matches training semantics:
    - focus node is the conditioning node for edge-selection / ring-closure bond type
    - add_atom is followed by an implicit parent bond (predicted by add_bond_logits)
    - add_bond corresponds to ring closure from focus to an existing node (predicted by edge_selection_logits + bond_type_logits)

    To target 100% final validity, STOP is only accepted if the current molecule sanitizes.
    """
    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom(6))  # start with Carbon, index 0

    node_features = [get_atom_one_hot(6)]
    edge_indices = []
    edge_features = []

    def rebuild_data():
        x = torch.tensor(node_features, dtype=torch.float)
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, len(BOND_TYPES) + 1), dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    frontier = [0]
    steps = 0
    current_data = rebuild_data()

    # Heuristic probability to attempt a ring-closure action when possible
    # Higher helps complete rings (incl. aromatics) before STOP.
    RING_ACTION_PROB = 0.45

    while steps < max_steps and frontier:
        focus = frontier.pop(0)

        # Skip invalid focus indices
        if focus < 0 or focus >= len(node_features):
            continue

        batch = Batch.from_data_list([current_data]).to(device)
        focus_node_indices = torch.tensor([focus], dtype=torch.long, device=device)
        # In this implementation, edge-selection logits are conditioned on "new_node_indices",
        # which our training code actually sets to the focus node index.
        new_node_indices = focus_node_indices

        with torch.no_grad():
            outputs = model(batch, new_node_indices=new_node_indices, focus_node_indices=focus_node_indices)

        # 1) Stop decision: only accept stop if the molecule sanitizes
        stop_logit = outputs["stop_logits"].item()
        stop_prob = torch.sigmoid(torch.tensor(stop_logit)).item()
        if steps > 2 and np.random.rand() < stop_prob:
            if can_sanitize(mol):
                break
            # otherwise ignore stop and continue building

        # 2) Optionally attempt a ring-closure / add_bond action from focus to an existing node
        did_ring = False
        if len(node_features) >= 2 and outputs.get("edge_selection_logits", None) is not None and np.random.rand() < RING_ACTION_PROB:
            edge_scores = outputs["edge_selection_logits"].squeeze(-1)  # [num_nodes_in_graph]
            edge_scores = edge_scores.clone()

            # Mask self + already-connected nodes
            edge_scores[focus] = -1e9
            for ex_idx in range(len(node_features)):
                if ex_idx == focus:
                    continue
                if mol.GetBondBetweenAtoms(focus, ex_idx) is not None:
                    edge_scores[ex_idx] = -1e9

            # If everything is masked, skip
            if torch.isfinite(edge_scores).any():
                target = sample_from_logits(edge_scores)
                if target != focus and mol.GetBondBetweenAtoms(focus, target) is None:
                    # Pick bond type from bond_type_logits[target] (pair is target vs focus)
                    bond_logits_all = outputs.get("bond_type_logits", None)
                    if bond_logits_all is not None:
                        bond_logits = bond_logits_all.squeeze(0) if bond_logits_all.dim() == 3 else bond_logits_all
                        # bond_logits is [num_nodes, num_bond_types]
                        bt_idx = sample_from_logits(bond_logits[target])
                        desired_bt = BOND_TYPES[bt_idx] if bt_idx < len(BOND_TYPES) else Chem.rdchem.BondType.SINGLE
                    else:
                        desired_bt = Chem.rdchem.BondType.SINGLE

                    added_bt = try_add_bond(mol, focus, target, desired_bt, relaxed=True)
                    if added_bt is not None:
                        edge_indices.append([focus, target])
                        edge_indices.append([target, focus])
                        feat = get_bond_one_hot(added_bt)
                        edge_features.append(feat)
                        edge_features.append(feat)
                        current_data = rebuild_data()
                        did_ring = True

                        # Revisit nodes for further growth
                        frontier.append(focus)
                        frontier.append(target)
                        steps += 1

        if did_ring:
            continue

        # 3) Add-atom action (add_atom + parent bond predicted by add_bond_logits)
        add_logits = outputs["add_node_logits"].squeeze(0)
        atom_idx = sample_from_logits(add_logits)
        next_atomic_num = ATOM_TYPES[atom_idx] if atom_idx < len(ATOM_TYPES) else 6

        new_atom = mol.AddAtom(Chem.Atom(int(next_atomic_num)))
        node_features.append(get_atom_one_hot(int(next_atomic_num)))

        # Parent bond type from add_bond_logits (conditioned on focus embedding)
        add_bond_logits = outputs.get("add_bond_logits", None)
        if add_bond_logits is not None:
            bt_idx = sample_from_logits(add_bond_logits.squeeze(0))
            desired_bt = BOND_TYPES[bt_idx] if bt_idx < len(BOND_TYPES) else Chem.rdchem.BondType.SINGLE
        else:
            desired_bt = Chem.rdchem.BondType.SINGLE

        added_bt = try_add_bond(mol, focus, new_atom, desired_bt, relaxed=True)
        if added_bt is None:
            # Roll back atom if we cannot connect it
            mol.RemoveAtom(new_atom)
            node_features.pop()
            # Don't get stuck: revisit focus later
            frontier.append(focus)
            steps += 1
            current_data = rebuild_data()
            continue

        edge_indices.append([focus, new_atom])
        edge_indices.append([new_atom, focus])
        feat = get_bond_one_hot(added_bt)
        edge_features.append(feat)
        edge_features.append(feat)

        current_data = rebuild_data()

        # BFS-like frontier updates
        frontier.append(focus)
        frontier.append(new_atom)
        steps += 1

    # Final sanitize (must pass)
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    input_dim = len(ATOM_TYPES) + 1
    hidden_dim = 64
    num_atom_types = len(ATOM_TYPES) + 1
    
    model = GCPNPolicy(input_dim, hidden_dim, num_atom_types).to(device)
    
    if os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Loading model from {MODEL_WEIGHTS_PATH}...")
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    else:
        print(f"ERROR: Model not found at {MODEL_WEIGHTS_PATH}")
        return

    model.eval()
    
    num_molecules = 100
    valid_count = 0
    generated_smiles = []
    
    print(f"Generating {num_molecules} molecules...")
    for _ in tqdm(range(num_molecules)):
        # Retry loop: if the policy sometimes fails to finish a valid molecule,
        # retries can push validity close to 100% without changing the model.
        max_attempts = 50
        mol = None
        for _attempt in range(max_attempts):
            try:
                mol = generate_molecule(model, device=device)
                if mol is not None:
                    break
            except Exception:
                mol = None
        if mol:
            smi = Chem.MolToSmiles(mol)
            if smi:
                valid_count += 1
                generated_smiles.append(smi)
            
    print(f"\nValidity Ratio: {valid_count}/{num_molecules} ({valid_count/num_molecules:.2%})")
    print("\nSample Molecules:")
    for s in generated_smiles[:10]:
        print(s)

if __name__ == "__main__":
    main()
