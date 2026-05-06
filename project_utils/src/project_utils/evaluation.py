"""Parallel graph evaluation and task management utilities.

This module provides tools for managing worker pools, filtering redundant tasks 
via a reward cache, and executing graph evaluations in parallel using 
multiprocessing or multithreading.
"""

import math
from collections import OrderedDict
import multiprocessing as mp
import multiprocessing.pool as mppool
from functools import lru_cache, partial

import psutil
import torch

from project_utils.graph import convert_bin_to_canonical_int


# number of worker threads to use in evaluate_gen function
@lru_cache
def n_jobs():
    """Determines the number of physical CPU cores available.

    Returns:
        int: The number of worker threads to use.
    """
    N_JOBS = psutil.cpu_count(logical=False)
    if N_JOBS is None:
        N_JOBS = int(input("UNABLE to determine the number of physical cores on the system. "
                            "Enter the number you want to use: "))
    print(f"Using {N_JOBS} workers for evaluation.")
    return N_JOBS


def filter_new_tasks(tasks, cache, rewards):
    """Filters tasks not present in cache while preserving order.

    Args:
        tasks (list): List of graph-like objects or hashable tasks.
        cache (dict): Mapping from task to computed reward.
        rewards (dict or list): Collection to be filled with known rewards.

    Returns:
        OrderedDict: Mapping from new tasks to lists of original indices.
    """
    new_tasks = OrderedDict()
    for i, task in enumerate(tasks):
        rew = cache.get(task)
        if rew is None:
            if task not in new_tasks:
                new_tasks[task] = []
            new_tasks[task].append(i)
        else:
            rewards[i] = rew

    return new_tasks

@lru_cache
def processes():
    """Initializes and caches a multiprocessing pool using 'spawn'.

    Returns:
        multiprocessing.pool.Pool: A process pool with n_jobs workers.
    """
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(n_jobs())
    return pool


def evaluate_gen_parallel(graphs, cache, eval_func, n_compute = None) -> tuple[int, torch.Tensor, int]:
    """Evaluates graph generations in parallel.

    Args:
        graphs (torch.Tensor): Array of shape (n, edges).
        cache (dict): Mapping from graph integer to reward.
        eval_func (callable): The reward evaluation function.
        n_compute (int, optional): Max number of new tasks to compute.

    Returns:
        tuple: (n_evaluated (int), rewards (torch.Tensor), new_graphs_cnt (int)).
    """
    n = graphs.shape[0]
    n_vertices = (1 + math.isqrt(1 + 8 * graphs.shape[1])) // 2
    if n_compute == None:
        n_compute = n

    rewards = torch.full((n,), -float('inf'), dtype=torch.float64)

    pool = processes()
    tasks = pool.map(convert_bin_to_canonical_int, graphs.detach().cpu().numpy())
    new_tasks = filter_new_tasks(tasks, cache, rewards)
    tasks_to_solve = list(new_tasks.keys())[:n_compute]

    worker = partial(eval_func, n_vertices=n_vertices)
    if getattr(eval_func, "multithreaded", False):
        with mppool.ThreadPool(n_jobs()) as pool:
            calculated_rewards = pool.map(worker, tasks_to_solve)
    else:
        pool = processes()
        calculated_rewards = pool.map(worker, tasks_to_solve)


    for calc_rew_pos, task in enumerate(tasks_to_solve):
        for in_pos in new_tasks[task]:
            rewards[in_pos] = calculated_rewards[calc_rew_pos]
        cache[task] = calculated_rewards[calc_rew_pos]

    n_evaluated = (rewards != -float('inf')).sum().item()
    # new_graphs_cnt is a number of new non-isomorphic graphs that appeared first in current generation
    new_graphs_cnt = len(new_tasks)

    return n_evaluated, rewards, new_graphs_cnt