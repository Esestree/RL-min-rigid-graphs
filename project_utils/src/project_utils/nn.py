"""
Module provides functionality for constructing, training, and performing inference with a neural network model.
"""

import math

import torch
import torch.nn as nn

from project_utils.henneberg_extensions import n_extensions, \
apply_actions, perform_actions
from project_utils.progress import Logger


class MLP(nn.Module):
    """
    A simple feedforward neural network with ReLU activations.
    """
    def __init__(self, layers_size: list):
        """
        Initializes the neural network.

        Args:
            layers_size (list): A list of integers specifying the sizes of each layer
                [input_dim, hidden1, hidden2, ..., output_dim].
        """

        super(MLP, self).__init__()
        layers = []
        sizes = layers_size[:-1]
        for i, size in enumerate(sizes[:-1]):
            layers.append(nn.Linear(size, sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[-1], layers_size[-1]))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def get_train_data(actions: torch.Tensor, n_vertices: int, base: list[int]=(0,)):
    """
    Create the training data for the graphs that were constructed using the given actions.

    Args:
        actions: The actions that were used for constructing the graphs.
                 They are in an array of shape (n, STEPS, n_extensions).
        n_vertices: number of vertices in graphs.
        base (list-like of int): initial subgraph in a construction sequence, given as 
            the indices of ones in the flattened adjacency matrix representation.
    Returns:
        xtrain: The training data for the neural network.
        ytrain: The labels for the training data.
    """
    n_edges = n_vertices * (n_vertices - 1) // 2
    v_base = (len(base) + 3) // 2
    n_steps = n_vertices - v_base
    n = actions.shape[0]

    # initialize xtrain
    xtrain = torch.zeros((n * (n_steps + 1), n_edges + n_steps))
    # start from minimally rigid graph H
    xtrain[:n, base] = 1

    # initialize ytrain
    ytrain = torch.zeros(n * n_steps, n_extensions(n_vertices))

    # for each step of the construction, apply the action on
    # previous partially constructed graphs and create the labels
    for i in range(n_steps):
        # one hot encoding for the vertex
        xtrain[i * n : (i + 1) * n, n_edges + i] = 1

        # copy current graphs to the next step
        xtrain[(i + 1) * n : (i + 2) * n, :n_edges] = xtrain[i * n : (i + 1) * n, :n_edges]

        # perform the actions on the next graphs
        labels = apply_actions(
            xtrain[(i + 1) * n : (i + 2) * n, :n_edges], n_vertices, actions[:, i, :], v_base + i
        )

        # create labels for current graphs - which actions should be performed to get the next graphs
        ytrain[torch.arange(i * n, (i + 1) * n), labels] = 1

    return xtrain[: n * n_steps, :], ytrain


class CrossEntropyLossWithEntropyRegularization(nn.Module):
    """Cross-Entropy loss with a dynamically decaying entropy regularization term."""

    def __init__(self, entropy_params):
        """Initializes the loss function with regularization parameters.

        Args:
            entropy_params (Any): Configuration object containing alpha, beta, 
                and coef_init attributes.
        """
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.entropy_alpha = entropy_params.alpha
        self.entropy_beta = entropy_params.beta
        self.entropy_coef_init = entropy_params.coef_init
        self.entropy_coef = None

    def coef_log_decay(self, gen):
        """Calculates the entropy coefficient based on logarithmic decay.

        Args:
            gen (int): The current generation index.

        Returns:
            float: The decayed entropy coefficient.
        """
        return (self.entropy_coef_init / 
                    (1 + self.entropy_alpha * math.log(1 + gen * math.e**(-self.entropy_beta))))

    def forward(self, input, target, gen):
        """Computes the total loss including cross-entropy and entropy regularization.

        Args:
            input (torch.Tensor): Predicted logits from the model.
            target (torch.Tensor): Ground truth labels.
            gen (int): The current generation index for coefficient decay.

        Returns:
            torch.Tensor: The combined loss value.
        """
        self.entropy_coef = self.coef_log_decay(gen)

        softmax = nn.Softmax(dim=1)
        probs = softmax(input)
        # Entropy calculation for the distribution
        entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-9), dim=1))

        loss = self.cross_entropy_loss(input, target) - self.entropy_coef * entropy
        return loss
    

def train_nn(xtrain, ytrain, model, criterion, optimizer, n_epochs = 1, logger = Logger(verbosity=1), **loss_kwargs):
    """
    Train the neural network for one epoch using the given training data, the criterion and the optimizer.

    Args:
        xtrain: The training data.
        ytrain: The labels for the training data.
        model: The neural network that is trained.
        criterion: The loss function that is used for training.
        optimizer: The optimizer that is used for training.
        logger: object that will save/print information.
    Returns:
        loss: The loss for the training data.
    """
    device = next(model.parameters()).device

    xtrain = xtrain.to(device)
    ytrain = ytrain.to(device)

    model.train()
    for i in range(n_epochs):
        pred = model(xtrain)
        loss = criterion(pred, ytrain, **loss_kwargs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.print(f"Loss after epoch {i + 1}:", round(loss.item(), 3))
    return round(loss.item(), 3)


def generate_generation(model, n: int, n_vertices: int, temperature: float = 1, base: list[int]=(0,),
                        logger = Logger(verbosity=1)):
    """
    Generate a new generation of graphs using the neural network.
    The graphs are generated by applying actions sampled from the neural network.

    Args:
        model: The neural network that is used for generating the graphs.
        n: The number of graphs to be generated.
        temperature: The temperature used in softmax.
        base (list-like of int): initial subgraph in a construction sequence, given as 
            the indices of ones in the flattened adjacency matrix representation.
        logger: object that will save/print information.

    Returns:
        graphs (torch.Tensor): The generated graphs in an array of shape (n, EDGES + VERTICES - 2).
        actions (torch.Tensor): The actions that were used for generating the graphs in an array of shape (n, STEPS, NACTIONS).
        stats (dict): saved statistics of the generation
    """
    n_edges = n_vertices * (n_vertices - 1) // 2
    v_base = (len(base) + 3) // 2
    n_steps = n_vertices - v_base

    device = next(model.parameters()).device

    graphs = torch.zeros((n, n_edges + n_steps))
    actions = torch.zeros((n, n_steps, n_extensions(n_vertices)))

    # start from minimally rigid graph H
    graphs[:, base] = 1

    model.eval()

    avg_predictions_probs = []

    # for each step of the construction, apply the predicted actions
    for i in range(n_steps):
        # encode current step using one hot encoding
        graphs[:, n_edges + i - 1] = 0
        graphs[:, n_edges + i] = 1

        # compute prediction
        with torch.no_grad():
            # get the predictions
            graphs = graphs.to(device)
            logits = model(graphs)
            graphs = graphs.to("cpu")

            # apply temperature
            logits = logits.to("cpu")
            logits = logits / temperature

            # apply softmax
            softmax = nn.Softmax(dim=1)
            pred = softmax(logits)

            avg_highest_prediction = torch.mean(torch.max(pred, dim=1)[0]).item()*100
            avg_predictions_probs.append(avg_highest_prediction)
            logger.print(f"Average highest prediction in step {i + 1}: {avg_highest_prediction:.2f}%")

            # zero out the predictions for actions that are not applicable
            pred = pred[:, :n_extensions(v_base + i + 1)]

        n = graphs.shape[0]
        # initialize arrays for storing performed actions and masks for which the actions were performed

        # what actions were sucesfully performed
        performed_actions = torch.zeros(n, dtype=torch.int64)

        # mask for which graphs the actions were performed
        performed_actions_mask = torch.zeros(n, dtype=torch.bool)
        
        # sample actions from the predictions - sampling K samples without replacement from distribution with K elements
        possible_actions = torch.multinomial(pred, n_extensions(v_base + i + 1), replacement=False)

        # while an action was not performed on all graphs
        counter = 0
        while performed_actions_mask.sum() != n:
            # actions that are going to be tried to be performed
            current_actions = possible_actions[~performed_actions_mask, counter]

            # store the actions that will be performed
            performed_actions[~performed_actions_mask] = current_actions

            # perform the actions on the graphs
            new_graphs, new = perform_actions(
                graphs[~performed_actions_mask], n_vertices, current_actions, v_base + i
            )

            # store the new graphs
            graphs[~performed_actions_mask] = new_graphs

            # update the mask
            performed_actions_mask[~performed_actions_mask] = new

            counter += 1

        # store the actions that were performed in the format of one hot encoding from all possible actions
        actions[torch.arange(actions.shape[0]), i, performed_actions] = 1

    # weights for prediction certainty through steps. Last step is the most important. 
    # Weights are assigned as w_1 = w, w_2 = 2w, w_3=4w, ... and sum of all w_i should be 1.
    step_importance_weights = torch.tensor([1/(2**len(avg_predictions_probs)-1) * 2**i 
                                            for i in range(len(avg_predictions_probs))])
    avg_predictions_probs = torch.tensor(avg_predictions_probs)
    stats = {'avg_prediction_certainty': torch.sum(step_importance_weights * avg_predictions_probs).item()}

    return graphs, actions, stats


def get_survival_graphs_actions(generation, actions, best_rewards_graphs, n_survive):
    """
    Return surviving graphs and their corresponding actions used for constructing them.

    Args:
        generation: The set of graphs in the current generation in an array of shape (n, EDGES) or (n, EDGES + STEPS).
        actions: The actions that were used for constructing the graphs in an array of shape (n, STEPS, NACTIONS).
        best_rewards_graphs: The rewards of the graphs in the current generation in an array of shape (n,) sorted
        in descending order.
        n_survive: how many graphs will survive to the next generation.

    Returns:
        survival_generation: The graphs that survived to the next generation.
        survival_actions: The actions that were used for constructing the surviving graphs.
    """
    survival_rewards = best_rewards_graphs[:n_survive]

    survival_actions = actions[survival_rewards, :]
    survival_generation = generation[survival_rewards, :]

    return survival_generation, survival_actions
