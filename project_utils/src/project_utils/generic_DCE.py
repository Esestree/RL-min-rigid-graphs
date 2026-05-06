"""Generic deep cross entropy method.

This module implements the Deep Cross Entropy (DCE) method, providing utilities 
for population management, statistical reporting, and the core training loop 
to optimize graph structures based on surrogate and main reward functions.
"""
from time import time

import torch
from omegaconf import DictConfig
import mlflow

from project_utils.evaluation import evaluate_gen_parallel
from project_utils.progress import save_tensors, save_best_graph, Logger, ModelHandler
from project_utils.graph import convert_networkx_to_adj_idx
from project_utils.nn import generate_generation, get_survival_graphs_actions, \
get_train_data, train_nn, CrossEntropyLossWithEntropyRegularization


def save_population(path, saved_surrogate_rewards, saved_main_rewards, saved_graphs, logger = Logger(verbosity=1)):
    """Saves the current population's rewards and graph structures to disk.

    Args:
        path (Path): Directory path where tensors will be saved.
        saved_surrogate_rewards (torch.Tensor): Tensor of surrogate rewards.
        saved_main_rewards (torch.Tensor): Tensor of main rewards.
        saved_graphs (torch.Tensor): Tensor of graph structures (bool).
        logger (Logger, optional): Logger instance for status updates.
    """
    save_tensors(
        (path / "saved_surrogate_rewards.pt", saved_surrogate_rewards),
        (path / "saved_main_rewards.pt", saved_main_rewards),
        (path / "saved_graphs.ptc", saved_graphs)  # .ptc says to save bool tensor in a compressed format
    )
    logger.print(f"Graphs and rewards saved")


def report_stats(gen: int, surrogate_rewards, main_rewards, avg_training_reward, new_graphs_cnt, 
                 train_loss, entropy_coef, avg_prediction_certainty, logger=Logger(verbosity=1)):
    """Logs and prints statistical metrics for the current generation.

    Args:
        gen (int): Current generation index.
        surrogate_rewards (torch.Tensor): Rewards from the surrogate function.
        main_rewards (torch.Tensor): Rewards from the main evaluation function.
        avg_training_reward (float): Average reward of the training elite set.
        new_graphs_cnt (int): Count of unique non-isomorphic graphs found.
        train_loss (float): Loss value from the neural network training.
        entropy_coef (float): Current entropy regularization coefficient.
        avg_prediction_certainty (float): Weighted certainty of model predictions.
        logger (Logger, optional): Logger instance for printing.
    """

    def log_print(name, value):
        logger.print(f"{name}: {value}")
        if mlflow.active_run():
            mlflow.log_metric(name, value, step=gen + 1)

    best_surrogate_reward = torch.max(surrogate_rewards).item()
    avg_surrogate_reward = round(torch.mean(surrogate_rewards).item(), 2)
    log_print("Highest surrogate reward", best_surrogate_reward)
    log_print("Average surrogate reward", avg_surrogate_reward)

    best_main_reward = torch.max(main_rewards).item()
    avg_main_reward = round(torch.mean(main_rewards).item(), 2)
    log_print("Highest main reward", best_main_reward)
    log_print("Average main reward", avg_main_reward)
    
    log_print("Average training reward", avg_training_reward)
    log_print("Count of new non-isomorphic graphs", new_graphs_cnt)
    log_print("Training loss", train_loss)
    log_print("Entropy coefficient", round(entropy_coef, 3))
    # avg_prediction_certainty is a sum of highest predictions in each step, weighted by importance.
    # Weights are assigned as w_1 = w, w_2 = 2w, w_3=4w, ... and sum of all w_i is 1.
    log_print("Average prediction certainty in percents", round(avg_prediction_certainty, 2))

    logger.print("\n")


def run_dce(cfg: DictConfig, main_reward_fn, surrogate_reward_fn, model, base_graph, saves_path) -> None:
    """Executes the Deep Cross Entropy optimization loop.

    Args:
        cfg (DictConfig): Configuration parameters for training and problem setup.
        main_reward_fn (callable): Expensive reward function for ground truth.
        surrogate_reward_fn (callable): Fast reward function for initial ranking.
        model (torch.nn.Module): Neural network used for graph generation.
        base_graph (nx.Graph): Starting graph structure for the generation process.
        saves_path (Path): Directory for saving models and population data.
    """
    logger = Logger(verbosity=cfg.general.verbosity)

    n_vertices = cfg.problem.n_vertices
    n_gen = cfg.training.n_gen
    gen_size = cfg.training.gen_size
    n_edges = n_vertices * (n_vertices - 1) // 2

    print(f"Training {n_gen} generations of "
          f"{gen_size} graphs with {n_vertices} vertices.\n")

    # number of graphs used for training, survival
    n_survive = max(int(cfg.training.survival_rate * gen_size), 1)
    n_train = max(int(cfg.training.train_best * gen_size), 1)

    # number of graphs to be evaluated with the expensive reward function after
    # their ranking was decided with fast surrogate reward function
    n_main_reward_compute = max(int(cfg.training.main_reward_compute_ratio * gen_size), n_train)

    # graphs and reward from all generations for saving
    saved_graphs = torch.zeros(n_gen, gen_size, n_edges, dtype=torch.bool)
    saved_surrogate_rewards = torch.zeros(n_gen, gen_size, dtype=torch.float64)
    saved_main_rewards = torch.zeros(n_gen, gen_size, dtype=torch.float64)

    # cache for evaluating the graphs
    main_rewards_cache = dict()
    surrogate_rewards_cache = dict()

    # initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
    loss_fn = CrossEntropyLossWithEntropyRegularization(cfg.entropy)
    
    # class that saves model's weights to the disk
    model_handler = ModelHandler(saves_path / "model_weights.pth")

    # no survivals before the first generation
    survival_generation, survival_actions = torch.empty(0), torch.empty(0)

    base_graph = convert_networkx_to_adj_idx(base_graph, n_vertices)

    for gen in range(n_gen):
        logger.print(f"GENERATION: {gen + 1}")

        # generate new set of graphs using the neural network
        generation, actions, generation_stats = generate_generation(model, gen_size - survival_generation.shape[0], 
                                                  n_vertices, base=base_graph, logger=logger)
        logger.print(f"Graphs generated")

        # add the survival graphs from previous iteration
        generation = torch.concat([generation, survival_generation], dim=0)
        actions = torch.concat([actions, survival_actions], dim=0)

        # evaluate all the graphs with fast approximation function
        eval_start = time()
        _, surrogate_rewards, new_graphs_cnt = evaluate_gen_parallel(
                        generation[:, :n_edges], surrogate_rewards_cache, surrogate_reward_fn)
        logger.print(f"Graphs evaluated with surrogate_reward_fn in {(time() - eval_start):.3f}s")
        best_surrogate_reward_graphs = torch.argsort(surrogate_rewards, descending=True)

        # use slow main reward function to evaluate only part of the graphs with the highest 
        # surrogate rewards
        eval_start = time()
        n_main_evaluated, main_rewards, _ = evaluate_gen_parallel(
                            generation[best_surrogate_reward_graphs, :n_edges], 
                            main_rewards_cache, main_reward_fn, n_main_reward_compute)
        logger.print(f"Graphs evaluated with reward in {(time() - eval_start):.3f}s")
        best_main_reward_graphs = torch.argsort(main_rewards, descending=True)

        # find elites for training
        best_graphs = best_surrogate_reward_graphs[best_main_reward_graphs[:n_train]]

        # create data for training
        xtrain, ytrain = get_train_data(actions[best_graphs, :, :], n_vertices, base=base_graph)

        # train the neural network
        train_loss = train_nn(xtrain, ytrain, model, loss_fn, optimizer, 
                            cfg.model.n_epochs, logger=logger, gen=gen)
        logger.print(f"Neural network trained")

        # find the graphs that will survive to the next generation
        survival_generation, survival_actions = get_survival_graphs_actions(
            generation, actions, best_graphs, n_survive
        )

        # save important statics, metrics, objects
        saved_main_rewards[gen, best_surrogate_reward_graphs] = main_rewards
        saved_surrogate_rewards[gen, :] = surrogate_rewards
        saved_graphs[gen, :, :] = generation[:, :n_edges]
        
        # save results once in artifact_saving_frequency generations
        if (gen + 1) % cfg.general.artifact_saving_frequency == 0:
            save_population(saves_path, saved_surrogate_rewards, saved_main_rewards, saved_graphs, logger)

        save_best_graph(generation[best_graphs[0], :n_edges], 
                        saves_path / "saved_best_graph.txt", logger=logger)

        model_handler.save_model(model)

        # print the statistics of the current generation and save to mlflow
        avg_training_reward = round(torch.mean(main_rewards[best_main_reward_graphs[:n_train]]).item(), 2)
        report_stats(gen, surrogate_rewards, main_rewards[best_main_reward_graphs[:n_main_evaluated]], avg_training_reward, 
                     new_graphs_cnt, train_loss, loss_fn.entropy_coef, generation_stats['avg_prediction_certainty'], logger)
        
    # save the total progress to a file
    save_population(saves_path, saved_surrogate_rewards, saved_main_rewards, saved_graphs, logger)
