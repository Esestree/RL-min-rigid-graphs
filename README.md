# Learning Minimally Rigid Graphs with High Realization Counts

Source code for the experiments in the paper "Learning Minimally Rigid Graphs with High Realization Counts" (Oleksandr Slyvka, Jan Rubeš, Rodrigo Alves, Jan Legerský). Use this toolkit to search for minimally rigid graphs with rare combinatorial features (e.g. maximized realization counts (planar/spherical), maximized number of NAC-colorings)

## Repository Structure

* **`project_utils/`**
A core reusable package providing the implementation of the Deep Cross-Entropy (DCE) method, neural network architecture, reward functions and helper functions

* **`experiments/`**
This directory manages problem-specific logic and configuration settings.
    * **`impls/`**: Targeted implementations for maximizing the number of realizations in **planar** and **spherical** contexts, as well as optimizing for **NAC colorings**.
    * **`parameters_config.yaml`**: A configuration file (Hydra/OmegaConf) managing all hyperparameters and problem constraints.
    * **`run.py`**: The primary entry point script used to initialize seeds, configure logging, and launch experiments.

* **`best_graphs_vis.ipynb`**
A visualization notebook created to display and check the highest reward graph structures generated during algorithm runs.


## Installation

All dependencies (including package versions) are listed in `requirements.txt` and can be installed into a virtual environment (venv) or Conda environment using pip:

`pip install -r requirements.txt`

The installation of the [`lnumber` library](https://github.com/jcapco/lnumber) may fail if the C++ **GMP** library is missing. This must be installed beforehand:

`sudo apt install libgmp-dev`

Finally, install the local `project_utils` package:

`pip install -e project_utils`


## Usage

Execute the following commands from within the `experiments` directory:

### Maximize Planar Realizations
`python3 run.py --config-name parameters_config.yaml impl=max_plane problem.n_vertices=<N>`

### Maximize Spherical Realizations
`python3 run.py --config-name parameters_config.yaml impl=max_sphere problem.n_vertices=<N>`

### Maximize NAC Colorings
`python3 run.py --config-name parameters_config.yaml impl=max_nac problem.n_vertices=<N>`

**Example:**
`python3 run.py --config-name parameters_config.yaml impl=max_plane problem.n_vertices=10`

You can also change other hyperparameters (see `parameters_config.yaml`)

### Monitoring Progress
You can monitor the algorithm's progress in the terminal or via the **MLflow UI**. To launch the UI, run the following command in the `experiments` folder:

`mlflow ui`

After the UI starts, navigate to the provided localhost URL and port in your browser. Each problem has a corresponding experiment folder located in the top-left corner of the dashboard. After clicking on the run name, you will find three main tabs: **Overview**, **Model metrics**, and **Artifacts**. The first displays the used hyperparameters, the second shows metrics from each generation, and the third allows you to download the model weights, graphs, and their corresponding rewards (e.g. realizations count).

You can stop the run with SIGINT (Ctrl+C), after any generation, all artifacts and plots will be preserved.
