"""
Module for functions related to creation and application of Hennenberg's extensions.
"""
from functools import cache

import torch

from project_utils.graph import get_edge_id


@cache
def generate_all_extensions(n_vertices: int):
    """
    Generate all possible extensions for the given number of vertices.
    The extensions are represented as tuples of vertices and possibly the edge to be deleted in 1-extension.
    The extensions are divided into 0-extensions and 1-extensions.
    0-extensions are represented as (i, j, -1, -1) where i and j are the vertices to which the new vertex is connected.
    1-extensions are represented as (i, j, k, l) where i, j, k are the vertices to which the new vertex is connected
    and l represents the edge that is deleted. For l = 0, the edge between i and j is deleted,
    for l = 1, the edge between j and k is deleted and for l = 2, the edge between i and k is deleted.

    The ALL_EXTENSIONS list is sorted by the maximum vertex used in the extension.

    Args:
        vertices: The number of vertices in the graph.
    Returns:
        all_extensions: All extensions for the given number of vertices.
    """
    # for creating a graph with n vertices, only n-1 vertices are used in the extensions
    # (now new vertex is connected to the last vertex)
    vertices_lower = n_vertices - 1

    # generate all 0-extensions
    zero_extensions = []

    # for all combinations of two vertices, create the 0-extension
    for i in range(vertices_lower):
        for j in range(i + 1, vertices_lower):
            zero_extensions.append((i, j, -1, -1))

    # generate all 1-extensions
    one_extensions = []

    # for all combinations of three vertices, create the corresponding 1-extensions
    for i in range(vertices_lower):
        for j in range(i + 1, vertices_lower):
            for k in range(j + 1, vertices_lower):
                one_extensions.append((i, j, k, 0))
                one_extensions.append((i, j, k, 1))
                one_extensions.append((i, j, k, 2))

    all_extensions = zero_extensions + one_extensions
    all_extensions.sort(key=max)

    return all_extensions

@cache
def generate_all_extensions_as_tensor(n_vertices: int):
    """
    Args:
        n_vertices: The number of vertices in the graph.
    Returns:
        extensions (torch.Tensor): All extensions for the given number of vertices.
    """
    extensions = torch.tensor(generate_all_extensions(n_vertices))
    return extensions

@cache
def n_extensions(n_vertices: int):
    """
    Args:
        n_vertices: The number of vertices in the graph.
    Returns:
        nextensions: the number of all possible extensions for the graph
    """
    nextensions = len(generate_all_extensions(n_vertices))
    return nextensions


def apply_one_extensions(graphs: torch.Tensor, n_vertices: int, actions: torch.Tensor, vertex: int):
    """
    Apply 1-extensions on the given graphs.
    The 1-extensions are applied only to graphs to which it is possible.

    Args:
        graphs: The graphs to which the 1-extensions are applied.
                They are in the vector representation.
                The shape is (n, edges).
        actions: The actions that are applied to the graphs.
                 They are in array of shape (n, 4).
                 The first three elements are the three vertices to which the new vertex is connected.
                 The last element is from the set {0, 1, 2} and it indicates which edge is to be deleted.
                 0 is the edge between first two vertices,
                 1 is the edge between the second and third vertex and
                 2 is the edge between the first and third vertex.
                 (i, j, k, l) means that the new vertex is connected to vertices i, j, k and
                 l represents the edge that is deleted.
        vertex: The new vertex that is being added to all the graphs.

    Returns:
        graphs: The graphs after the 1-extensions are applied.
        not_applicable: A mask that indicates for which graphs the 1-extensions were not applied.
    """
    # masks for edges that are to be deleted
    first_edge = actions[:, -1] == 0
    second_edge = actions[:, -1] == 1
    third_edge = actions[:, -1] == 2

    n = graphs.shape[0]

    # initialize the array for edges to be deleted
    edges_to_del = torch.zeros((n, 2), dtype=torch.int64)

    # get the edges to be deleted using the created masks and the actions
    edges_to_del[first_edge, 0] = actions[first_edge, 0]
    edges_to_del[first_edge, 1] = actions[first_edge, 1]
    edges_to_del[second_edge, 0] = actions[second_edge, 1]
    edges_to_del[second_edge, 1] = actions[second_edge, 2]
    edges_to_del[third_edge, 0] = actions[third_edge, 0]
    edges_to_del[third_edge, 1] = actions[third_edge, 2]

    # get the edges to be deleted id - the index of the edge in the vector representation of the graph
    edges_to_del_id = get_edge_id(edges_to_del, n_vertices)

    # get the mask for which the 1-extension is applicable - for which there exists the edge to be deleted
    applicable = graphs[torch.arange(graphs.size(0)), edges_to_del_id] == 1
    # filter out the unapplicable graphs
    applicable_graphs = graphs[applicable, :]
    edges_to_del_id = edges_to_del_id[applicable]
    edges_to_del = edges_to_del[applicable, :]
    actions = actions[applicable, :]

    # edges to be added
    edges = torch.zeros((3 * actions.shape[0], 2))
    # edges from
    edges[:, 0] = actions[:, :3].reshape(-1)
    # edges to
    edges[:, 1] = vertex

    # get their id
    edges_id = get_edge_id(edges, n_vertices)
    # for every graph, 3 edges are added in 1-extension -> reshape the edges_id
    # so that each row is for one graph
    edges_id = edges_id.reshape(-1, 3)

    # apply the 1-extension
    # add the edges
    applicable_graphs[torch.arange(applicable_graphs.size(0)), edges_id[:, 0]] = 1
    applicable_graphs[torch.arange(applicable_graphs.size(0)), edges_id[:, 1]] = 1
    applicable_graphs[torch.arange(applicable_graphs.size(0)), edges_id[:, 2]] = 1
    # remove the edge
    applicable_graphs[torch.arange(applicable_graphs.size(0)), edges_to_del_id] = 0

    # update the original array
    graphs[applicable, :] = applicable_graphs

    return graphs, ~applicable


def apply_zero_extensions(graphs: torch.Tensor, n_vertices: int, actions: torch.Tensor, vertex: int):
    """
    Apply 0-extensions on the given graphs.

    Args:
        graphs: The graphs to which the 0-extensions are applied.
                They are in the vector representation.
                The shape is (n, edges).
        actions: The actions that are applied to the graphs.
                 They are in array of shape (n, 2).
                 The elements are the two vertices to which the new vertex is connected.
        vertex: The new vertex that is being added to all the graphs.

    Returns:
        graphs: The graphs after the 0-extensions are applied.
    """
    n = graphs.shape[0]

    # the edges to be added
    edges = torch.zeros((2 * n, 2))
    # edges from
    edges[:, 0] = actions[:, :2].reshape(-1)
    # edges to
    edges[:, 1] = vertex

    # get their id - the index of the edge in the vector representation of the graph
    edges_id = get_edge_id(edges, n_vertices)

    # for every graph, 2 edges are added in 0-extension -> reshape the edges_id so
    # that each row is for one graph
    edges_id = edges_id.reshape(-1, 2)

    # add the edges to the graphs
    graphs[torch.arange(graphs.size(0)), edges_id[:, 0]] = 1
    graphs[torch.arange(graphs.size(0)), edges_id[:, 1]] = 1

    return graphs


def perform_actions(graphs: torch.Tensor, n_vertices: int, actions: torch.Tensor, vertex: int):
    """
    Perform the actions on the given graphs (not in place). Returns the new graphs and the performed actions.

    Args:
        graphs: The graphs to which the actions are applied.
            They are in the vector representation in an array of shape (n, edges).
        actions: The actions that are applied to the graphs.
            They are in array of shape (n,). Each element is the index of the action.
        vertex: The new vertex that is being added to all the graphs.
    Returns:
        graphs: The graphs after the actions are applied.
            They are in the vector representation in an array of shape (n, edges).
        performed_actions: A mask for graphs that represents whether an action was performed,
            because there is a possibility that some 1-extensions were not applicable.
            An array of shape (n,).
    """
    # convert actions id to the action actions
    # (i, j, ?, -1) means that 0-extension is applied and new vertex is connected to i and j
    # (i, j, k, l) means that 1-extension is applied and new vertex is connected to i, j, k and
    # l represents the edge that is deleted
    # for more info see apply_one_extensions and apply_zero_extensions functions
    all_extensions = torch.tensor(generate_all_extensions(n_vertices))
    actions = all_extensions[actions]

    # mask for 1-extensions
    one_extensions = actions[:, -1] != -1

    # apply the 1-extensions and update the graphs
    new_graphs, not_applicable = apply_one_extensions(
        graphs[one_extensions, :], n_vertices, actions[one_extensions], vertex
    )
    graphs[one_extensions, :] = new_graphs

    # mask for 0-extensions
    zero_extensions = ~one_extensions

    # apply the 0-extensions and update the graphs
    new_graphs = apply_zero_extensions(
        graphs[zero_extensions, :], n_vertices, actions[zero_extensions, :2], vertex
    )
    graphs[zero_extensions, :] = new_graphs

    n = graphs.shape[0]

    # mask for which graphs the actions were performed
    performed_extension = torch.zeros(n, dtype=torch.bool)
    performed_extension[one_extensions] = ~not_applicable
    performed_extension[zero_extensions] = True

    return graphs, performed_extension


def apply_actions(graphs: torch.Tensor, n_vertices: int, actions: torch.Tensor, vertex: int):
    """
    Apply the actions on the given graphs (in place) and return the actions that were performed.

    Args:
        graphs: The graphs to which the actions are applied.
            They are in the vector representation in an array of shape (n, edges).
        actions: The actions that are applied to the graphs.
            The actions with the highest values are applied first.
            They are in array of shape (n, NACTIONS).
        vertex: The new vertex that is being added to all the graphs.
    Returns:
        performed_actions: The actions that were performed on the graphs.
            They are in an array of shape (n,).
            Each element is the index of the performed action.
    """
    # filter out the actions that do not use edges that are between vertices that are not in the graph yet
    # if vertex 10 is being added, 0-extension that connects the new vertex
    # to vertices 12 and 2 is not applicable, because vertex 12 is not in the graph yet
    actions = actions[:, : n_extensions(vertex + 1)]

    # sort the actions in descending order
    actions_sorted = torch.argsort(actions, dim=1, descending=True)
    n = graphs.shape[0]

    # initialize arrays for storing performed actions and masks for which the actions were performed
    performed_actions_mask = torch.zeros(n, dtype=torch.bool)
    performed_actions = torch.zeros(n, dtype=torch.int64)
    counter = 0

    # while an action was not performed on all graphs
    while performed_actions_mask.sum() != n:
        # actions that are going to be tried to be performed
        performed_actions[~performed_actions_mask] = actions_sorted[
            ~performed_actions_mask, counter
        ]

        # try to perform the actions on the graphs
        new_graphs, new = perform_actions(
            graphs[~performed_actions_mask],
            n_vertices,
            actions_sorted[~performed_actions_mask, counter],
            vertex,
        )

        # update the graphs and the mask
        graphs[~performed_actions_mask] = new_graphs
        performed_actions_mask[~performed_actions_mask] = new
        counter += 1

    return performed_actions


def generate_random_graphs(n: int, n_vertices: int, base: list[int]=(0,)):
    """
    Generate random minimally rigid graphs on n_vertices vertices from base graph using extensions.

    Args:
        n: The number of graphs to generate.
        n_vertices: Total number of vertices in the final graphs.
        base (list-like of int): initial subgraph in a construction sequence, given as 
            the indices of ones in the flattened adjacency matrix representation.

    Returns:
        (graphs, actions): 
            - graphs (Tensor): Binary tensor of shape (n, NUM_EDGES + NUM_STEPS),
                         representing each graph as a flattened edge vector with step encoding.
            - actions (Tensor): Tensor of shape (n, NUM_STEPS, ACTION_DIM),
                          where each step contains the encoded extension action (e.g., [i, j, k]).

    """
   
    n_edges = n_vertices * (n_vertices - 1) // 2

    v_base = (len(base) + 3) // 2
    n_steps = n_vertices - v_base

    # initialize the arrays
    actions = torch.zeros((n, n_steps, n_extensions(n_vertices)))
    graphs = torch.zeros((n, n_edges + n_steps))

    # start from minimally rigid graph H
    graphs[:, base] = 1

    for i in range(n_steps):
        # generate random value for every action for every graph. For each graph the action with highest value
        # will be applied in current step. If it is not possible to apply the chosen action, the action
        # with second highest value will be considered, and so on
        actions_probs = torch.rand(n, n_extensions(n_vertices))
        performed_actions = apply_actions(graphs[:, :n_edges], n_vertices, actions_probs, v_base + i)
        actions[torch.arange(actions.size(0)), i, performed_actions] = 1

    return graphs, actions
