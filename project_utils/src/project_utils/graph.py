"""
Utility functions for working with graph representations, encodings, and edge mappings.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    import torch


def convert_networkx_to_adj_idx(graph: nx.Graph, n_vertices):
    """Converts a NetworkX graph to a flat indices tensor of existing edges.

    Args:
        graph (nx.Graph): The input NetworkX graph.
        n_vertices (int): Total number of vertices to ensure in the graph.

    Returns:
        torch.Tensor: Indices of non-zero entries in the upper triangle.
    """
    import torch
    graph = graph.copy()
    graph.add_nodes_from(range(n_vertices))
    
    adj = nx.to_numpy_array(graph)
    adj = torch.from_numpy(adj).bool()
    i, j = torch.triu_indices(n_vertices, n_vertices, offset=1)
    triangle = adj[i, j]

    return torch.nonzero(triangle, as_tuple=True)[0]


def convert_bin_to_int(graph: torch.Tensor) -> int:
    """
    Convert the binary vector representation of the graph to an
    integer representation that is used by the lnumber library.

    Args:
        graph (torch.Tensor): Binary vector of the upper triangle.

    Returns:
        int: Integer representation of the graph.
    """
    res = 0
    for val in graph:
        res = (res << 1) | int(val)
    return res


def convert_bin_to_canonical_int(graph: np.array) -> int:
    """Converts a binary vector to an integer reperesentation of a canonical adjacency matrix using pynauty.

    Args:
        graph (np.array): Binary vector of the upper triangle.

    Returns:
        int: Canonical integer representation of the graph.
    """
    import math
    import pynauty as nauty

    m = graph.shape[0]
    n = int((1 + math.isqrt(1 + 8 * m)) // 2)
    
    adj = np.zeros((n, n), dtype=graph.dtype)
    iu = np.triu_indices(n, 1)
    adj[iu[0], iu[1]] = graph
    adj = adj + adj.T
    
    adj_dict = {i: np.where(adj[i] == 1)[0].tolist() for i in range(n)}
    G = nauty.Graph(n, directed=False, adjacency_dict=adj_dict)
    
    perm = np.array(nauty.canon_label(G))
    
    canonical_adj = adj[perm][:, perm]
    canonical_upper = canonical_adj[iu[0], iu[1]]

    canonical_int = convert_bin_to_int(canonical_upper) 
    return canonical_int


def convert_int_to_networkx(graph: int, n_vertices: int) -> nx.Graph:
    """Converts an integer representation of an Laman graph into a NetworkX graph.

    Args:
        graph (int): Integer whose binary representation corresponds to the upper triangle
                   of the adjacency matrix (excluding diagonal).
        n_vertices (int): Number of vertices in the graph.

    Returns:
        nx.Graph: The corresponding NetworkX graph with nodes labeled 0 to n_vertices - 1.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))

    bits = bin(graph)[2:]
    idx = 0
    bits = bits.zfill(n_vertices * (n_vertices - 1) // 2)

    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if bits[idx] == '1':
                G.add_edge(i, j)
            idx += 1
    return G

def convert_networkx_to_int(graph: nx.Graph) -> int:
    """Converts a NetworkX graph to an integer representation.
    Args:
        graph (nx.Graph): An undirected NetworkX graph with nodes labeled 0 to n - 1.

    Returns:
        int: An integer whose binary representation corresponds to the upper triangle
             of the adjacency matrix (excluding the diagonal).
    """
    n = graph.number_of_nodes()
    bits = 0
    # start with most significant bit
    bit_pos = n * (n - 1) // 2 - 1
    
    # Generate edges in lex order: (0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2,n-1)
    for i in range(n):
        for j in range(i + 1, n):
            if graph.has_edge(i, j):
                bits |= (1 << bit_pos)
            bit_pos -= 1

    return bits


def get_edge_id(edges, n_vertices):
    """Calculates edge IDs based on their index in the vector representation.

    Return the edge id for the given edges.
    Edge id is the index of the edge in the vector representation of the graph.

    Args:
        edges (torch.Tensor): The edges for which the edge id is to be returned.
            They are in an array of shape (n, 2). One row is one edge.
        n_vertices (int): Total number of vertices in the graph.

    Returns:
        torch.Tensor: Tensor of edge IDs.
    """
    import torch
    # sort the edges so that the smaller vertex is first
    # and the larger vertex is second
    with torch.no_grad():
        edges, _ = torch.sort(edges, dim=1, descending=False)

    res = torch.zeros(edges.shape[0], dtype=torch.int32)
    all_edges = ((i, j) for i in range(n_vertices) for j in range(i+1, n_vertices))
    for i, edg in enumerate(all_edges):
        mapping = (edges[:, 0] == edg[0]) & (edges[:, 1] == edg[1])
        res[mapping] = i

    return res