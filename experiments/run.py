"""Orchestrator for Hydra-based experiments with MLflow tracking.

This script serves as the main entry point for running experiments. It 
handles parameter flattening for MLflow logging, sets global seeds, and 
dynamically loads implementations based on the provided configuration.
"""

import os
import importlib
import random

import mlflow
import hydra
from omegaconf import OmegaConf, DictConfig


def log_config(cfg):
    """Flattens and logs the Hydra configuration to MLflow.

    Args:
        cfg (DictConfig): The OmegaConf configuration object to log.
    """
    flat_cfg = OmegaConf.to_container(cfg, resolve=True)

    def flatten_dict(d, parent_key=None, sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    flat_params = flatten_dict(flat_cfg)

    print("Running with params:")
    for k, v in flat_params.items():
        print(f"{k}={v}")
    
    # Log all flattened parameters at once
    mlflow.log_params(flat_params)


@hydra.main(version_base=None, config_path=".")
def main(cfg: DictConfig):
    """Main execution block for Hydra that manages MLflow runs.

    Args:
        cfg (DictConfig): Configuration object provided by Hydra.
    """
    mlflow.set_experiment(cfg.experiment_name)
    run_name = f"{cfg.problem.n_vertices}n,{cfg.impl}"

    with mlflow.start_run(run_name=run_name):
        log_config(cfg)

        from project_utils.progress import set_seed
        set_seed(cfg.general.seed)

        # Dynamically import and run selected implementation
        module_path = f"impls.{cfg.impl}"
        try:
            implementation_module = importlib.import_module(module_path)
            implementation_module.main(cfg)
        except ModuleNotFoundError as e:
            print("FAILED to import module:", e)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", eval, use_cache=True)
    OmegaConf.register_new_resolver("random", random.randint, use_cache=True)

    # All output files will be saved to directory with this script
    # Get the directory where this script resides
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Override current dir
    os.chdir(script_dir)
    print(f"Hydra outputs and Mlflow mlruns will be stored in: {os.getcwd()}")

    main()