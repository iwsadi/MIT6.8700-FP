import torch
import random
import networkx as nx
from rdkit import Chem
from torch_geometric.data import Data

# Allowed atom types and bond types for one-hot encoding
ATOM_TYPES = [6, 7, 8, 9, 15, 16, 17, 35, 53] # C, N, O, F, P, S, Cl, Br, I
# 0 is reserved for 'other' if needed, or we just map everything else to a bucket
BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, 
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

def get_atom_one_hot(atomic_num):
    # One-hot encoding for atom type
    # Returns a list of floats
    one_hot = [0.0] * (len(ATOM_TYPES) + 1)
    if atomic_num in ATOM_TYPES:
        idx = ATOM_TYPES.index(atomic_num)
        one_hot[idx] = 1.0
    else:
        one_hot[-1] = 1.0 # Unknown/Other
    return one_hot

def get_bond_one_hot(bond_type):
    # One-hot encoding for bond type
    one_hot = [0.0] * (len(BOND_TYPES) + 1)
    if bond_type in BOND_TYPES:
        idx = BOND_TYPES.index(bond_type)
        one_hot[idx] = 1.0
    else:
        one_hot[-1] = 1.0 # Unknown/Other
    return one_hot

class MoleculeTrajectory:
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        if self.mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        # Sanitize to ensure proper bond types etc
        Chem.SanitizeMol(self.mol)

    def generate_steps(self):
        """
        Generates a trajectory of (State, Action) pairs using randomized BFS.
        """
        mol = self.mol
        num_atoms = mol.GetNumAtoms()
        
        # Pick a random starting atom
        atom_indices = list(range(num_atoms))
        start_atom_idx = random.choice(atom_indices)
        
        # Data structures for the constructing graph
        # We will build lists of features and edges incrementally
        constructed_node_features = [] # List of one-hot vectors
        constructed_edge_indices = []  # List of [src, dst]
        constructed_edge_features = [] # List of one-hot vectors
        
        # Mapping from original molecule atom idx to constructed graph node idx
        mol_to_graph = {}
        graph_to_mol = {}
        
        # BFS Queue
        queue = []
        visited_mol_indices = set()
        
        trajectory = []
        
        # --- Step 0: Initialize with the first atom ---
        # State: Empty graph
        # Action: Add the first atom
        
        current_data = self._build_data_object(constructed_node_features, 
                                               constructed_edge_indices, 
                                               constructed_edge_features)
        
        start_atom = mol.GetAtomWithIdx(start_atom_idx)
        start_atomic_num = start_atom.GetAtomicNum()
        
        # Initial atom has no bond
        action = {
            'type': 'add_atom',
            'atom_type': start_atomic_num,
            'bond_type': None
        }
        trajectory.append((current_data, action))
        
        # Apply Step 0
        mol_to_graph[start_atom_idx] = 0
        graph_to_mol[0] = start_atom_idx
        constructed_node_features.append(get_atom_one_hot(start_atomic_num))
        visited_mol_indices.add(start_atom_idx)
        queue.append(0) # Store graph index in queue
        
        # --- BFS Loop ---
        while queue:
            # Pop from front (BFS)
            u_graph_idx = queue.pop(0)
            u_mol_idx = graph_to_mol[u_graph_idx]
            u_atom = mol.GetAtomWithIdx(u_mol_idx)
            
            # Get neighbors and shuffle for randomness
            neighbors = list(u_atom.GetNeighbors())
            random.shuffle(neighbors)
            
            for v_atom in neighbors:
                v_mol_idx = v_atom.GetIdx()
                bond = mol.GetBondBetweenAtoms(u_mol_idx, v_mol_idx)
                bond_type = bond.GetBondType()
                
                if v_mol_idx in visited_mol_indices:
                    # Neighbor already visited. Check if edge exists in constructed graph.
                    v_graph_idx = mol_to_graph[v_mol_idx]
                    
                    # Check if edge (u, v) exists
                    # Since we add edges bidirectionally, checking one direction is enough
                    edge_exists = False
                    for edge in constructed_edge_indices:
                        if (edge[0] == u_graph_idx and edge[1] == v_graph_idx) or \
                           (edge[0] == v_graph_idx and edge[1] == u_graph_idx):
                            edge_exists = True
                            break
                    
                    if not edge_exists:
                        # Ring closure or just connecting to an already visited node
                        # State: Current graph
                        # Action: Add bond to v
                        
                        # We include a 'focus_node' indicator? 
                        # The prompt requirement doesn't strictly ask for it, 
                        # but implied by 'target_node_idx' relative to *something*.
                        # We assume the policy knows the 'focus' is implicit or we can add it to features.
                        # For now, I will stick to the requested output format.
                        
                        current_data = self._build_data_object(constructed_node_features, 
                                                               constructed_edge_indices, 
                                                               constructed_edge_features,
                                                               focus_node_idx=u_graph_idx)
                        
                        action = {
                            'type': 'add_bond',
                            'target_node_idx': v_graph_idx,
                            'bond_type': bond_type
                        }
                        trajectory.append((current_data, action))
                        
                        # Apply Action
                        self._add_edge(constructed_edge_indices, constructed_edge_features, 
                                       u_graph_idx, v_graph_idx, bond_type)
                        
                else:
                    # Neighbor not visited. Add new atom.
                    
                    # 1. Add Atom
                    # We pass 'bond_type' here so the model can learn "Add Atom X with Bond Type Y"
                    current_data = self._build_data_object(constructed_node_features, 
                                                           constructed_edge_indices, 
                                                           constructed_edge_features,
                                                           focus_node_idx=u_graph_idx)
                    action = {
                        'type': 'add_atom',
                        'atom_type': v_atom.GetAtomicNum(),
                        'bond_type': bond_type # Future-proofing for "Add Atom + Bond" joint prediction
                    }
                    trajectory.append((current_data, action))
                    
                    # Apply Action 1
                    v_graph_idx = len(constructed_node_features)
                    mol_to_graph[v_mol_idx] = v_graph_idx
                    graph_to_mol[v_graph_idx] = v_mol_idx
                    constructed_node_features.append(get_atom_one_hot(v_atom.GetAtomicNum()))
                    visited_mol_indices.add(v_mol_idx)
                    queue.append(v_graph_idx)
                    
                    # 2. Add Bond (connect new atom to u)
                    # Note: Now v is in the graph (disconnected from u). 
                    # The focus should effectively shift to v temporarily to connect back? 
                    # Or stay at u and connect to v?
                    # The prompt says: "target_node_idx: The index of the node to bond with".
                    # If we stay at u, target is v.
                    # If we move focus to v (implicit in add_atom?), target is u.
                    # Usually "Add Atom" appends to the list. 
                    # Let's assume focus stays at u for this loop.
                    
                    current_data = self._build_data_object(constructed_node_features, 
                                                           constructed_edge_indices, 
                                                           constructed_edge_features,
                                                           focus_node_idx=u_graph_idx)
                    
                    action = {
                        'type': 'add_bond',
                        'target_node_idx': v_graph_idx, # Connect u to the newly added v
                        'bond_type': bond_type
                    }
                    trajectory.append((current_data, action))
                    
                    # Apply Action 2
                    self._add_edge(constructed_edge_indices, constructed_edge_features, 
                                   u_graph_idx, v_graph_idx, bond_type)
        
        # --- Final Stop Action ---
        current_data = self._build_data_object(constructed_node_features, 
                                               constructed_edge_indices, 
                                               constructed_edge_features)
        action = {'type': 'stop'}
        trajectory.append((current_data, action))
        
        return trajectory

    def _add_edge(self, edge_indices, edge_features, u, v, bond_type):
        # Add undirected edge (represented as two directed edges)
        edge_indices.append([u, v])
        edge_indices.append([v, u])
        
        feat = get_bond_one_hot(bond_type)
        edge_features.append(feat)
        edge_features.append(feat)

    def _build_data_object(self, node_features, edge_indices, edge_features, focus_node_idx=None):
        if not node_features:
            # Empty graph case
            # Ensure edge_attr is present even for empty graphs to satisfy PyG collation
            return Data(x=torch.zeros((0, len(ATOM_TYPES)+1)), 
                        edge_index=torch.zeros((2,0), dtype=torch.long),
                        edge_attr=torch.zeros((0, len(BOND_TYPES)+1), dtype=torch.float))
            
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Optionally add a feature for "focus node" if provided
        # I'll append it as the last feature column if requested, 
        # but to stick to the prompt's "one-hot for atom type" strictness, 
        # I will store it in a separate attribute 'focus_node' in the Data object.
        # This is cleaner and allows the user to decide how to use it.
        
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, len(BOND_TYPES)+1), dtype=torch.float)
            
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        if focus_node_idx is not None:
            data.focus_node = focus_node_idx
            
        return data

if __name__ == "__main__":
    # Example usage
    smiles = "CCO"
    traj_gen = MoleculeTrajectory(smiles)
    trajectory = traj_gen.generate_steps()
    
    print(f"SMILES: {smiles}")
    print(f"Trajectory Length: {len(trajectory)}")
    for i, (state, action) in enumerate(trajectory):
        print(f"Step {i}:")
        print(f"  Nodes: {state.num_nodes}, Edges: {state.num_edges}")
        if hasattr(state, 'focus_node'):
            print(f"  Focus Node: {state.focus_node}")
        print(f"  Action: {action}")

