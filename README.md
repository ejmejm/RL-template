# Reinforcement Learning Implementation Template

This project is meant to provide a quick and easy template to implement or develop a wide variety of RL algorithms. 
When starting a new RL project, I would always find myself rewriting the same code and having to spend unecessary time 
debugging code I had already written dozens of times before. This project provides a simple and adjustable implementation
of policy gradient in both 1D and 2D environments. Environments are simulated over multiple processes using mpi4py
making the process very efficient.

## Prerequisites

First you need to install a working version of MPI. Many different versions exist, but I personally recommend OpenMPI,
Instructions can be found [here](https://mpi4py.readthedocs.io/en/stable/appendix.html#building-mpi).

The template has only been tested with Python3.6, so versions 3.6 and above should all work. Other versions may also work, but are not guaranteed.
Clone the repository then run a pip install on the `requirements.txt` file to get the necessary modules installed.

```
git clone https://github.com/ejmejm/RL-template.git
cd ./RL-template
pip install -r requirements.txt
```

\* It is recommended that you use Tensorflow with GPU support for faster training times, but the demos can also
be run on just a CPUs.

## Getting Started

To get started, two demos are provided using OpenAI gym's [CartPole](https://gym.openai.com/envs/CartPole-v0/) (1D)
and [Breakout](https://gym.openai.com/envs/Breakout-v0/) (2D) environments.

To run the CartPole demo on 4 cores run `mpiexec -n 4 python3 run_cart_pole.py`

To run the Breakout demo on 4 cores run `mpiexec -n 4 python3 run_breakout.py`

By changing the `-n` flag, you can adjust the number of cores used to run the program.

## Implementation

The implementation is meant to be simple and easily understandable while still being maximally efficient.
Simulation of environments is done over multiple processes in parallel using [mpi4py](https://mpi4py.readthedocs.io/en/stable/).
To achieve this, the main network is copied to all processes, and weights are synced between all networks after each training epoch.
Below is a list of individual files and corresponding functions.

* **`models.py`** - Implements a `BaseModel` class that can used as a basis to implement most policy value based RL algorithms and sync weights with MPI.
Implements a `OneDimModel` that extends `BaseModel` and implements vanilla policy gradient in a 1D environment with a discrete action-space.
Implements a `TwoDimModel` that extends `BaseModel` and implements vanilla policy gradient in a 2D environment with a discrete action-space.

* **`utils.py`** - Defines utility functions for logging, filtering 2D observations, and calculating GAEs + dicounted rewards.

* **`run_cart_pole.py` and `run_breakout.py`** - Both implement worker functions that define how an environment is simulated. They then run a main
loop that consists of running environments in parallel, gathering + formatting the data, using the data to train the main network, and then propagating
the weights from the main network to all copied networks.

## Authors

* **Edan Meyer** - [GitHub Profile](https://github.com/ejmejm)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
