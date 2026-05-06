"""
Entry point for running the Deep Cross Entropy (DCE) graph optimization.

This script initializes the model architecture (MLP or GIN) based on the 
provided configuration, sets up the reward functions, and starts the 
optimization proces.
"""

from pathlib import Path
from omegaconf import DictConfig
import networkx as nx
import torch

from project_utils.generic_DCE import run_dce
from project_utils.rewards import NumSphere, RealizationsUpperBound
from project_utils.nn import MLP
from project_utils.gnn import GIN
from project_utils.henneberg_extensions import n_extensions

def main(cfg: DictConfig):
    """Initializes the environment and executes the DCE optimization loop.

    Args:
        cfg (DictConfig): Configuration object containing model parameters, 
            problem constraints, and training settings.
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Running on device: {device}")

    n_vertices = cfg.problem.n_vertices
    n_edges = n_vertices * (n_vertices - 1) // 2

    # from which subgraph to construct all the graphs
    base_graph = nx.Graph(cfg.problem.base_graph)

    # initialize neural network
    if cfg.model.type == 'MLP':
        # number of steps in the graph construction
        n_steps = n_vertices - base_graph.number_of_nodes()
        layers = [n_edges + n_steps] + cfg.model.nn_size + [n_extensions(n_vertices)]
        model = MLP(layers).to(device)
    elif cfg.model.type == 'GIN':
        layers = cfg.model.nn_size
        model = GIN(layers, n_vertices, cfg.model.n_GIN_layers).to(device)
    else:
        raise ValueError("Model type should be GIN or MLP")

    if cfg.model.load_weights_from is not None:
        model.load_state_dict(torch.load(cfg.model.load_weights_from, weights_only=True))

    saves_path = Path("tmp_saves").resolve()

    run_dce(cfg, NumSphere(), RealizationsUpperBound(), model, base_graph, saves_path)
