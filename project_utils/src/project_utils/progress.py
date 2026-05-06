"""Module for saving progress of an algorithm run. This module provides utilities for logging, setting seeds for reproducibility, 
handling model weights, and efficient saving/loading of tensors—including a custom compressed format for boolean tensors.
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import mlflow
import random
import numpy as np
from pathlib import Path

from project_utils.graph import convert_bin_to_int

def set_seed(seed):
    """Sets seeds for random, numpy, and torch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger:
    """A simple verbosity-controlled wrapper for the print function."""

    def __init__(self, verbosity):
        """Initializes the Logger with a specific verbosity level.

        Args:
            verbosity (int): Verbosity level (higher values mean more output).
        """
        self.verbosity = verbosity

    def print(self, *args, **kwargs):
        """Prints messages if verbosity is greater than zero."""
        if self.verbosity > 0:
            print(*args, **kwargs)


def save_bool_tensor(tensor: torch.Tensor, path: str):
    """Compresses and saves a boolean tensor by packing bits into bytes.

    Args:
        tensor (torch.Tensor): Tensor of dtype=torch.bool.
        path (str): File path for saving the compressed data.
    """
    assert tensor.dtype == torch.bool, "Input tensor must be of dtype torch.bool"

    # Get total number of elements in tensor
    n = tensor.numel()
    # Save original shape to restore after loading
    shape = tensor.shape

    tensor = tensor.flatten()

    # If padding is needed, append zeros (False) to tensor
    pad_len = (8 - (n % 8)) % 8
    if pad_len > 0:
        tensor = torch.cat([tensor, torch.zeros(pad_len, dtype=torch.bool)])

    # Reshape tensor into rows of 8 bools for packing into one byte
    tensor = tensor.view(-1, 8)

    # Convert every 8 bools to uint8
    masks = 2 ** torch.arange(7, -1, -1, dtype=torch.uint8)
    packed = (tensor.to(torch.uint8) * masks).sum(dim=1)
    packed = packed.to(dtype=torch.uint8)

    # Prepare data dictionary containing necessary information to restore the tensor back
    data = {
        "packed": packed,
        "shape": shape,
        "length": n
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(data, path)


def load_bool_tensor(path: str) -> torch.Tensor:
    """Loads and unpacks a compressed boolean tensor.

    Args:
        path (str): Path to the compressed file.

    Returns:
        torch.Tensor: The restored boolean tensor.
    """
    data = torch.load(path, weights_only=True)

    packed = data["packed"]
    shape = data["shape"]
    length = data["length"]

    masks = 2 ** torch.arange(7, -1, -1, dtype=torch.uint8)

    # Unpack 8 bits from one byte into 1D tensor
    unpacked = ((packed.unsqueeze(1) & masks) > 0).reshape(-1)

    # Remove padding bits by slicing to original length, then reshape to original tensor shape
    unpacked = unpacked[:length].reshape(shape)

    return unpacked


def save_tensors(*tensors: tuple[str, torch.Tensor]):
    """Saves multiple tensors, automatically compressing if file extension is .ptc.

    Args:
        *tensors: Variable number of tuples in the format (filename, tensor).
    """
    for path, tensor in tensors:
        if path.suffix == ".ptc":
            assert tensor.dtype == torch.bool, "Compressed tensor files (.ptc) expect bool tensors"
            save_bool_tensor(tensor, path)
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(tensor, path)
    
    if mlflow.active_run():
        for path, _ in tensors:
            mlflow.log_artifact(path)


def load_tensors(*files: str) -> tuple[torch.Tensor]:
    """Loads multiple tensors, handling .ptc compression automatically.

    Args:
        *files: Variable number of file paths as strings.

    Returns:
        tuple[torch.Tensor]: A tuple of loaded tensors.
    """
    loaded_tensors = []
    for filename in files:
        if filename.endswith(".ptc"):
            # For compressed bool tensors
            tensor = load_bool_tensor(filename)
        else:
            tensor = torch.load(filename, weights_only=True)
        loaded_tensors.append(tensor)

    return tuple(loaded_tensors)

def save_best_graph(best_graph, path, logger=Logger(verbosity=1)):
    """Converts the best graph to an integer and saves it to a file.

    Args:
        best_graph (torch.Tensor): Binary representation of the graph.
        path (str): File path for saving.
        logger (Logger, optional): Logger for output.
    """
    best_graph = convert_bin_to_int(best_graph)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as file:
        file.write(str(best_graph))
    if mlflow.active_run():
        mlflow.log_artifact(path)
    logger.print("Highest reward graph saved")


class ModelHandler:
    """Manages model state dictionary saving and MLflow logging."""

    def __init__(self, save_path):
        """Initializes the handler with a save path.

        Args:
            save_path (str): Destination path for model weights.
        """
        self.save_path = save_path
    
    def save_model(self, model):
        """Saves the model state dict and logs it as an MLflow artifact.

        Args:
            model (torch.nn.Module): The model to save.
        """
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.save_path)
        if mlflow.active_run():
            mlflow.log_artifact(self.save_path)
