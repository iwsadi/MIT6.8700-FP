import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Batch

# Import bond types for bond-type head dimensions
try:
    from create_trajectory import BOND_TYPES
except ImportError:
    # Fallback for relative import when running within GCPN package
    from .create_trajectory import BOND_TYPES

class GCPNPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_atom_types, num_layers=3, conv_type='GCN', dropout=0.1):
        super(GCPNPolicy, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_atom_types = num_atom_types
        self.conv_type = conv_type
        self.dropout = dropout
        
        # 1. Encoder: 3-layer GCN/GAT backbone
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_projs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            if conv_type == 'GCN':
                self.convs.append(GCNConv(in_channels, hidden_dim))
            elif conv_type == 'GAT':
                # GAT usually has heads, here we assume 1 head or concat=False/projected
                self.convs.append(GATConv(in_channels, hidden_dim, heads=1))
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")
            self.norms.append(nn.LayerNorm(hidden_dim))
            # Residual projection if dimensions mismatch
            self.res_projs.append(nn.Linear(in_channels, hidden_dim) if in_channels != hidden_dim else nn.Identity())
                
        # 2. Policy Heads
        
        # mlp_stop: Graph embedding -> prob of termination
        # Output 1 scalar (logit for probability)
        self.mlp_stop = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # mlp_add_node: Graph embedding -> logits for next atom type
        self.mlp_add_node = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_atom_types)
        )
        
        # New head: mlp_add_bond
        # Predicts the bond type for the newly added atom (connecting to the focus node)
        # Input: Graph embedding (representing the focus node context implicitly via pooling, 
        # or we could use focus node embedding explicitly, but graph embedding is simpler for now).
        # Better: Use focus node embedding? The focus node is the "parent".
        # Let's use Graph Embedding + Focus Node Embedding?
        # For simplicity and to match existing patterns, let's just use Graph Embedding.
        # It should contain info about the focus node if we had a mechanism to highlight it, 
        # but standard global_mean_pool loses "focus".
        # 
        # CRITICAL FIX: We need to know WHICH node is the focus node to predict the bond type correctly?
        # Actually, in the 'Add Node' step, we are adding a node *to the focus node*.
        # So the bond is between (New Node) and (Focus Node).
        # We don't have the New Node embedding yet (it doesn't exist).
        # We only have the Focus Node embedding.
        # So we should predict: f(h_focus) -> bond_type_logits.
        
        self.mlp_add_bond = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(BOND_TYPES) + 1)
        )
        
        # mlp_edge_selection: (h_new, h_existing) -> score
        # We'll concatenate embeddings: input dim is 2 * hidden_dim
        self.mlp_edge_selection = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Bond type head: predict bond class for a (new_node, existing_node) pair
        self.mlp_bond_type = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(BOND_TYPES) + 1)  # +1 for 'other'
        )

    def forward(self, batch_data, new_node_indices=None, focus_node_indices=None):
        """
        Args:
            batch_data: torch_geometric.data.Batch object
            new_node_indices: Optional LongTensor of shape [batch_size]. 
                              Indices of the 'newly added node' in the whole batch.
                              If provided, computes edge selection scores.
            focus_node_indices: Optional LongTensor of shape [batch_size].
                                Indices of the 'focus node' (parent) in the whole batch.
                                Used to predict the initial bond type for the added atom.
                              
        Returns:
            stop_logits: [batch_size, 1]
            add_node_logits: [batch_size, num_atom_types]
            add_bond_logits: [batch_size, num_bond_types] (Prediction for connection to parent)
            node_embeddings: [num_nodes_in_batch, hidden_dim]
            edge_selection_logits: [num_nodes_in_batch, 1] or None if new_node_indices not provided
        """
        x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch
        
        # 1. Encoder
        h = x
        for conv, norm, res_proj in zip(self.convs, self.norms, self.res_projs):
            h_in = h
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            # Residual
            h = h + res_proj(h_in)
        
        node_embeddings = h
        
        # 2. Aggregation
        # Global mean pool to get graph-level embedding
        # shape: [batch_size, hidden_dim]
        batch_size = batch_data.num_graphs
        graph_embedding = global_mean_pool(node_embeddings, batch, size=batch_size)
        
        # 3. Policy Heads
        
        # Stop prediction
        stop_logits = self.mlp_stop(graph_embedding)
        
        # Add Node prediction
        add_node_logits = self.mlp_add_node(graph_embedding)
        
        # Add Bond prediction (connection to parent)
        add_bond_logits = None
        if focus_node_indices is not None:
            # Extract embedding of focus nodes
            # shape: [batch_size, hidden_dim]
            # Use gather or direct indexing. direct indexing is fine if focus_node_indices are global.
            focus_node_embeddings = node_embeddings[focus_node_indices]
            add_bond_logits = self.mlp_add_bond(focus_node_embeddings)
            
        # Edge Selection (if applicable)
        edge_selection_logits = None
        bond_type_logits = None
        if new_node_indices is not None:
            # new_node_indices should be the global indices in the batch corresponding to the new nodes
            
            # Extract embeddings of new nodes
            # shape: [batch_size, hidden_dim]
            new_node_embeddings = node_embeddings[new_node_indices]
            
            # We need to broadcast these back to all nodes in the respective graphs
            # 'batch' gives us the graph index for each node
            # shape: [num_nodes_in_batch, hidden_dim]
            expanded_new_node_embeddings = new_node_embeddings[batch]
            
            # Concatenate: [h_target, h_new_broadcasted]
            # shape: [num_nodes_in_batch, 2 * hidden_dim]
            pair_embeddings = torch.cat([node_embeddings, expanded_new_node_embeddings], dim=1)
            
            # Calculate scores
            # shape: [num_nodes_in_batch, 1]
            edge_selection_logits = self.mlp_edge_selection(pair_embeddings)
            # Bond type logits per pair: [num_nodes_in_batch, num_bond_types]
            bond_type_logits = self.mlp_bond_type(pair_embeddings)
            
            # Masking logic could be applied externally (e.g. don't connect to itself)
            
        return {
            "stop_logits": stop_logits,
            "add_node_logits": add_node_logits,
            "add_bond_logits": add_bond_logits,
            "node_embeddings": node_embeddings,
            "edge_selection_logits": edge_selection_logits,
            "bond_type_logits": bond_type_logits,
        }

