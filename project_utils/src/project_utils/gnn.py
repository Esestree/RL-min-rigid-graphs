"""Module for GIN-based graph neural networks.

This module implements a Graph Isomorphism Network (GIN) designed to score 
potential Henneberg extensions (0-extensions and 1-extensions) on graphs, 
incorporating various node features such as LDP, clustering coefficients, 
and step embeddings.
"""

from functools import lru_cache

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.transforms import AddLaplacianEigenvectorPE, LocalDegreeProfile
from project_utils.henneberg_extensions import generate_all_extensions_as_tensor


@lru_cache(maxsize=1024 * 4)
def compute_lap_pe_cached(edge_index_key: tuple, num_nodes: int, k: int):
    """Computes and caches Laplacian Positional Encodings for a graph.

    Args:
        edge_index_key (tuple): Tuple of edges representing the graph structure.
        num_nodes (int): Total number of nodes in the graph.
        k (int): Number of eigenvectors to compute.

    Returns:
        torch.Tensor: The computed Laplacian positional encodings.
    """
    edge_index_list = list(edge_index_key)
    edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long).T

    data = tg.data.Data(edge_index=edge_index_tensor, num_nodes=num_nodes)
    
    lap_pe = AddLaplacianEigenvectorPE(k=k, attr_name=None, is_undirected=True)
    data = lap_pe(data)
    return data.x


class GIN(torch.nn.Module):
    """Graph Isomorphism Network for scoring graph extensions."""
    def __init__(self, layers, n_vertices, n_gin_layers):
        """Initializes the GIN model with specified layer dimensions.

        Args:
            layers (list): List of dimensions [hidden, out, scorer_hidden].
            n_vertices (int): Maximum number of vertices in the graphs.
            n_gin_layers (int): Number of GIN convolution layers.
        """
        super().__init__()

        # node features
        self.n_ldp = 5
        self.n_clustering_coef = 1
        self.step_emb_dim = 2
        self.step_emb = nn.Embedding(23, self.step_emb_dim)
        # self.n_lap_eigenvectors = n_vertices - 1

        self.n_node_features = self.n_ldp + self.n_clustering_coef + self.step_emb_dim

        self.n_vertices = n_vertices
        gin_conv_hidden_dim = layers[0]
        gin_conv_out = layers[1]
        ext_sc_hidden_dim = layers[2]

        self.convs = nn.ModuleList([
            tg.nn.GINConv(nn.Sequential(
                nn.Linear(self.n_node_features if i == 0 else gin_conv_hidden_dim, gin_conv_hidden_dim),
                nn.ReLU(),
                nn.Linear(gin_conv_hidden_dim, gin_conv_out if i == n_gin_layers - 1 else gin_conv_hidden_dim)
            ), train_eps=True) for i in range(n_gin_layers)
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(gin_conv_out if i == n_gin_layers - 1 else gin_conv_hidden_dim) for i in range(n_gin_layers)])

        self.extension_scorer = nn.Sequential(
            nn.Linear(2 * gin_conv_out + 5, ext_sc_hidden_dim),
            nn.ReLU(),
            nn.Linear(ext_sc_hidden_dim, ext_sc_hidden_dim),
            nn.ReLU(),
            nn.Linear(ext_sc_hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes a batch of graphs and scores their possible extensions.

        Args:
            x (torch.Tensor): Input batch of graphs of shape (n, edges), where edges are given as 
            upper triangle of adjacency matrix.

        Returns:
            torch.Tensor: Padded scores for each possible extension per graph.
        """
        data = self.graphs_to_pyg_batch(x)
        data = self.add_node_features(data)
        out = data.x
        edge_index = data.edge_index
        batch = data.batch

        for conv, bn in zip(self.convs, self.batch_norms):
            out = conv(out, edge_index)
            out = bn(out)
            out = F.relu(out)

        extensions_batch, extension_counts = self.make_graphwise_extension_batch(out, batch, edge_index)
        scores = self.extension_scorer(extensions_batch)

        scores_per_graph = list(torch.split(scores, extension_counts))

        for i in range(len(scores_per_graph)):
            scores_per_graph[i] = scores_per_graph[i].squeeze(dim=-1)  # shape: [num_extensions_i]

        max_len = max(score.size(0) for score in scores_per_graph)
        padded_scores = torch.full((len(scores_per_graph), max_len), fill_value=-1e9, device=scores.device)

        for i, score in enumerate(scores_per_graph):
            padded_scores[i, :score.size(0)] = score

        return padded_scores


    
    def graphs_to_pyg_batch(self, graphs, compute_on_device="cpu"):
        """Converts raw graph tensors to a PyTorch Geometric batch.

        Args:
            graphs (torch.Tensor): Tensor representing a batch of graphs.
            compute_on_device (str): Device to perform initial processing on.

        Returns:
            tg.data.Batch: A processed PyG batch object.
        """
        device = next(self.parameters()).device
        graphs = graphs.to(compute_on_device)

        n_edges = self.n_vertices * (self.n_vertices - 1) // 2
        edge_bits = graphs[:, :n_edges] > 0.5

        i_list, j_list = [], []
        for i in range(self.n_vertices):
            for j in range(i + 1, self.n_vertices):
                i_list.append(i)
                j_list.append(j)
        pairs = torch.tensor([i_list, j_list])  # shape: [2, n_edges]

        pyg_graphs = []
        for g in range(len(graphs)):
            # Get indices of edges that exist in this graph
            mask = edge_bits[g]
            edge_index = pairs[:, mask]  # shape: [2, num_edges]

            # Make undirected
            edge_index_rev = edge_index[[1, 0], :]
            edge_index = torch.cat([edge_index, edge_index_rev], dim=1)

            num_nodes = len(torch.unique(edge_index[0]))
            x = torch.empty((num_nodes, 0), dtype=torch.float)


            pyg_graphs.append(tg.data.Data(x=x, edge_index=edge_index))

        return tg.data.Batch.from_data_list(pyg_graphs).to(device)


    def apply_lap_pe(self, batch_data, compute_on_device="cpu"):
        """Applies cached Laplacian Positional Encodings to a batch.

        Args:
            batch_data (tg.data.Batch): The input PyG batch.
            compute_on_device (str): Device for computation.

        Returns:
            tg.data.Batch: Batch with Laplacian PE features added.
        """
        device = next(self.parameters()).device
        batch_data = batch_data.to(compute_on_device)
        graphs = batch_data.to_data_list()

        processed_graphs = []
        for graph in graphs:
            k = min(graph.num_nodes - 1, self.n_lap_eigenvectors)

            edge_index_tuple = tuple(map(tuple, graph.edge_index.cpu().numpy().T))
            lap_pe = compute_lap_pe_cached(edge_index_tuple, graph.num_nodes, k)

            graph.x = torch.cat((graph.x, lap_pe), dim=1)
            padding = torch.zeros(len(graph.x), self.n_lap_eigenvectors - lap_pe.shape[1])
            graph.x = torch.cat((graph.x, padding), dim=1)

            processed_graphs.append(graph)

        return tg.data.Batch.from_data_list(processed_graphs).to(device)


    def apply_ldp(self, batch_data, compute_on_device="cpu"):
        """Adds Local Degree Profile features to the node features. Local degree profile adds 
        5 features: deg(v), mean(DN(v)), std(DN(v)), min(DN(v)), max(DN(v)).

        Args:
            batch_data (tg.data.Batch): The input PyG batch.
            compute_on_device (str): Device for computation.

        Returns:
            tg.data.Batch: Batch with LDP features added.
        """
        device = next(self.parameters()).device
        batch_data = batch_data.to(compute_on_device)

        ldp_transform = LocalDegreeProfile()
        batch_data = ldp_transform(batch_data)
        
        # scale degrees based on the property that the best graphs have degrees between 2 and 4
        batch_data.x[:, -5: -1] = (batch_data.x[:, -5:-1] - 2) / (4 - 2)

        return batch_data.to(device)

    def apply_clustering_coef(self, batch_data, compute_on_device="cpu"):
        """Adds clustering coefficient as a node feature.

        Args:
            batch_data (tg.data.Batch): The input PyG batch.
            compute_on_device (str): Device for computation.

        Returns:
            tg.data.Batch: Batch with clustering coefficient features added.
        """
        device = next(self.parameters()).device
        batch_data = batch_data.to(compute_on_device)

        G = tg.utils.to_networkx(batch_data, to_undirected=True)

        # Compute clustering coefficient per node {node: value}
        cc_dict = nx.clustering(G)

        cc = torch.tensor([cc_dict[i] for i in range(batch_data.num_nodes)],
                        dtype=torch.float, device=compute_on_device).view(-1, 1)
        
        batch_data.x = torch.cat([batch_data.x, cc], dim=-1)

        return batch_data.to(device)

    def add_step_embedding(self, batch_data):
        """Adds an embedding representing the current construction step.

        Args:
            batch_data (tg.data.Batch): The input PyG batch.

        Returns:
            tg.data.Batch: Batch with step embeddings added to node features.
        """
        node_counts = torch.bincount(batch_data.batch)  # [num_graphs]
        step_emb_graph = self.step_emb(node_counts)  # [num_graphs, emb_dim]
        step_emb_node = step_emb_graph[batch_data.batch]  # [total_nodes, emb_dim]

        batch_data.x  = torch.cat([batch_data.x , step_emb_node], dim=-1)

        return batch_data

    def add_node_features(self, batch_data):
        """Orchestrates the addition of all enabled node features.

        Args:
            batch_data (tg.data.Batch): The input PyG batch.

        Returns:
            tg.data.Batch: Batch with all computed node features and padding.
        """
        if hasattr(self, "n_ldp"):
            batch_data = self.apply_ldp(batch_data)
        if hasattr(self, "n_lap_eigenvectors"):
            batch_data = self.apply_lap_pe(batch_data)
        if hasattr(self, "n_clustering_coef"):
            batch_data = self.apply_clustering_coef(batch_data)
        if hasattr(self, "step_emb_dim"):
            batch_data = self.add_step_embedding(batch_data)
        
        padding = torch.zeros(len(batch_data.x), self.n_node_features - batch_data.x.shape[1],
                              device=batch_data.x.device)
        batch_data.x = torch.cat((batch_data.x, padding), dim=1)

        return batch_data


    def make_graphwise_extension_batch(self, node_features: torch.Tensor, batch: torch.Tensor, edge_index: torch.Tensor):
        """Creates a batch of embeddings for all possible graphs extensions.

        Args:
            node_features (torch.Tensor): Feature matrix of shape [total_nodes, feat_dim].
            batch (torch.Tensor): Batch assignment vector.
            edge_index (torch.Tensor): Graph connectivity.

        Returns:
            tuple: (extension_feats [total_exts, 2*feat_dim+5], extension_counts [List]).
        """

        device = node_features.device
        feat_dim = node_features.size(1)
        num_graphs = batch.max().item() + 1

        # Compute node offsets for each graph to convert local to global index
        node_counts = torch.bincount(batch, minlength=num_graphs)  # [num_graphs]
        node_ptr = torch.cat([torch.tensor([0], device=device), torch.cumsum(node_counts, dim=0)])  # [num_graphs + 1]

        # Generate list of extensions for each graph (list of Tensor[num_exts, 4])
        exts_list = [generate_all_extensions_as_tensor(num_nodes.item() + 1) for num_nodes in node_counts]

        # Record graph index of each extension
        graph_idx_list = []
        for g, exts in enumerate(exts_list):
            graph_idx_list.append(torch.full((exts.size(0),), g, dtype=torch.long))

        flat_exts = torch.cat(exts_list, dim=0).to(device) # [total_exts, 4]
        flat_graph_idx = torch.cat(graph_idx_list, dim=0).to(device) # [total_exts]
        
        v123 = flat_exts[:, :3]                 # [total_exts, 3]
        removed_edge = flat_exts[:, 3]          # [total_exts]

        # Convert local v1/v2/v3 to global node indices
        global_offsets = node_ptr[flat_graph_idx]               # [total_exts]
        v123_global = v123 + global_offsets.unsqueeze(1)        # [total_exts, 3]

        # Append zero vector as last row to node features
        zero_vector = torch.zeros(1, feat_dim, device=device)
        node_features_padded = torch.cat([node_features, zero_vector], dim=0)

        # index of zero vector
        last_index = node_features_padded.size(0) - 1 

        # Replace -1 with last_index
        v123_global[v123 == -1] = last_index

        # Gather node features
        v_feats = node_features_padded[v123_global.view(-1)]           # [total_exts * 3, feat_dim]
        v_feats = v_feats.view(-1, 3, feat_dim)                # [total_exts, 3, feat_dim]




        # add removed edge features to express what connection is removed. Take embedding of two disconnected nodes
        perms = torch.tensor([[0, 1], [1, 2], [0, 2]])
        removed_edge_features = torch.zeros(v_feats.shape[0], 2, feat_dim, device=device)

        for i, perm in enumerate(perms):
            removed_edge_features[removed_edge == i] = v_feats[removed_edge == i][:, perm]
        
        # apply permutation invariant function to combine node embeddings. Use sum
        removed_edge_features = removed_edge_features.sum(dim=1)
        removed_edge_features[removed_edge == -1] = 0.0




        # apply permutation invariant function to combine node embeddings. Use sum
        v_feats = v_feats.sum(dim=1)




        # One-hot encode E1a, E1b and E1c subclass of 1-extensions, like in
        # Explorations on the number of realizations of minimally rigid graphs
        # For invalid extension assign unique type. For 0-ext the type is also unique.
        # map edges to graphs
        edge2graph = batch[edge_index[0]]
        src = edge_index[0]
        tgt = edge_index[1].clone()
        node_counts_cum_sum = torch.cat((torch.tensor([0], device=device), torch.cumsum(node_counts, dim=0)))

        # replace the second vertex ID with its local node index in the graph
        tgt -= node_counts_cum_sum[edge2graph]
        adj_matrix = torch.zeros(batch.size(0), self.n_vertices, dtype=torch.bool, device=device)
        adj_matrix[src, tgt] = True
        
        # take all combinations of 2 vertices in extensions
        src = v123_global[:, [0, 1, 0]]
        tgt = v123[:, [1, 2, 2]]

        # find size of a subgraph which connects nodes in each extension 
        existent_edges = adj_matrix[src, tgt]
        sizes_of_extenions_subgraphs = existent_edges.sum(dim=1)

        # find extensions that cannot be performed
        ext_cannot_be_permofmed = torch.zeros(removed_edge.size(0), dtype=torch.bool, device=device)
        for i, perm in enumerate(perms):
            ext_cannot_be_permofmed[removed_edge == i] = ~existent_edges[removed_edge == i, i]

        sizes_of_extenions_subgraphs[ext_cannot_be_permofmed] = 0

        # for 0-extension the type is unique
        sizes_of_extenions_subgraphs[removed_edge == -1] = 4

        ext_subclasses = F.one_hot(sizes_of_extenions_subgraphs, num_classes=5).float()  # [total_exts, 5]
        



        # Concatenate features
        extension_feats = torch.cat([v_feats, removed_edge_features, ext_subclasses], dim=1)  # [total_exts, 2 * feat_dim + 5]
        # Count extensions per graph (for later splitting/masking)
        extension_counts = torch.bincount(flat_graph_idx, minlength=num_graphs).tolist()

        return extension_feats, extension_counts
