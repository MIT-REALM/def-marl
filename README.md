<div align="center">

# Def-MARL

[![Conference](https://img.shields.io/badge/RSS-Accepted-success)](https://mit-realm.github.io/def-marl/)

Jax official implementation of RSS2025 paper (Best Student Paper Award): [Songyuan Zhang](https://syzhang092218-source.github.io), [Oswin So](https://oswinso.xyz/), [Mitchell Black](https://www.blackmitchell.com/), [Zachary Serlin](https://zacharyserlin.com/) and [Chuchu Fan](https://chuchu.mit.edu): "[Solving Multi-Agent Safe Optimal Control with Distributed Epigraph Form MARL](https://mit-realm.github.io/def-marl/)". 

[Dependencies](#Dependencies) •
[Installation](#Installation) •
[Quickstart](#Quickstart) •
[Environments](#Environments) •
[Algorithms](#Algorithms) •
[Usage](#Usage)

</div>

<div align="center">
    <img src="./media/video/MPETarget.gif" alt="MPETarget" width="16.2%"/>
    <img src="./media/video/MPESpread.gif" alt="LidarLine" width="16.2%"/>
    <img src="./media/video/MPEFormation.gif" alt="VMASReverseTransport" width="16.2%"/>
    <img src="./media/video/MPELine.gif" alt="VMASWheel" width="16.2%"/>
    <img src="./media/video/MPECorridor.gif" alt="VMASWheel" width="16.2%"/>
    <img src="./media/video/MPEConnectSpread.gif" alt="VMASWheel" width="16.2%"/>
</div>

<div align="center">
    <img src="./media/img/def-marl.jpg" alt="Def-MARL Framework" width="100%"/>
</div>

## Dependencies

We recommend to use [CONDA](https://www.anaconda.com/) to install the requirements:

```bash
conda create -n defmarl python=3.10
conda activate defmarl
```

Then install the dependencies:
```bash
pip install -r requirements.txt
```

## Installation

Install Def-MARL:

```bash
pip install -e .
```

## Quickstart

To train a model on the `LidarSpread` environment, run:

```bash
python train.py --env LidarSpread --algo def-marl -n 3 --obs 3
```

To evaluate a model, run:

```bash
python test.py --path ./logs/LidarSpread/def-marl/seed0_xxxxxxxxxx
```

## Environments

### MPE

We provide the following environments with the [MPE](https://github.com/openai/multiagent-particle-envs) (Multi-Agent Particle Environment) simulation engine: `MPETarget`, `MPESpread`, `MPEFormation`, `MPELine`, `MPECorridor`, `MPEConnectSpread`. In MPE, agents, goals/landmarks, and obstacles are represented as particles. Agents can observe other agents or obstacles when they are within their observation range. Agents follow the double integrator dynamics.

<div align="center">
    <img src="media/img/MPETarget.jpg" width="16.2%" alt="MPETarget">
    <img src="media/img/MPESpread.jpg" width="16.2%" alt="MPESpread">
    <img src="media/img/MPEFormation.jpg" width="16.2%" alt="MPEFormation">
    <img src="media/img/MPELine.jpg" width="16.2%" alt="MPELine">
    <img src="media/img/MPECorridor.jpg" width="16.2%" alt="MPECorridor">
    <img src="media/img/MPEConnectSpread.jpg" width="16.2%" alt="MPEConnectSpread">
    <img src="media/img/env_legend.jpg" alt="Legend" width="100%">
</div>

- `MPETarget`: The agents need to reach their pre-assigned goals.
- `MPESpread`: The agents need to collectively cover a set of goals without having access to an assignment.
- `MPEFormation`: The agents need to spread evenly around a given landmark.
- `MPELine`: The agents need to form a line between two given landmarks.
- `MPECorridor`: The agents need to navigate through a narrow corridor and cover a set of given goals.
- `MPEConnectSpread`: The agents need to cover a set of given goals while maintaining connectivity.

### Other Environments

The code can also work with other simulation engines including [LidarEnv](https://github.com/MIT-REALM/gcbfplus/) and [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator). Examples using these environments can be found in our [DGPPO](https://github.com/MIT-REALM/dgppo/) repository.

### Custom Environments
It is very easy to create a custom environment by yourself! Inherit one of the existing environments, then define your reward function, graph connection, and dynamics, register the new environment in `env/__init__.py`, and you are good to go!

## Algorithms

We provide the following algorithms:

- `def-marl`: Distributed Epigraph Form Multi-agent Reinforcement Learning (Def-MARL).
- `informarl`: [MAPPO](https://github.com/marlbenchmark/on-policy) with GNN ([Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation](https://github.com/nsidn98/InforMARL/)).
- `informarl_lagr`: [MAPPO-Lagrangian](https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation) with GNN.

## Usage

### Train

To train the `<algo>` algorithm on the `<env>` environment with `<n_agent>` agents and `<n_obs>` obstacles, run:

```bash
python train.py --env <env> --algo <algo> -n <n_agent> --obs <n_obs>
```

The training logs will be saved in `logs/<env>/<algo>/seed<seed>_<timestamp>`. We provide the following flags:

#### Required Flags

- `--env`: Environment. 
- `--algo`: Algorithm.
- `-n`: Number of agents.
- `--obs`: Number of obstacles.

#### For algorithms

- `--cost-weight`: [For informarl] Weight of the cost term in the reward, default 0.0.
- `--lagr-init`: [For informarl_lagr] Initial value of the Lagrangian multiplier, default 0.78.
- `--lr-lagr`: [For informarl_lagr] Learning rate of the Lagrangian multiplier, default 1e-7.
- `--clip-eps`: Clip epsilon, default 0.25.
- `--coef-ent`: Entropy coefficient, default 0.01.

#### For environments

- `--full-observation`: Use full observation, default False.
- `--area-size`: Side length of the environment, default None.

#### Training options
- `--no-rnn`: Do not use RNN, default False. **Use this flag in the VMAS environments can accelerate the training**.
- `--n-env-train`: Number of environments for training, default 128. Decrease this number with `--batch-size` if the GPU memory is not enough.
- `--batch-size`: Batch size, default 16384.
- `--n-env-test`: Number of environments for testing (during training), default 32.
- `--log-dir`: Directory to save the logs, default `logs`.
- `--eval-interval`: Evaluation interval, default 1. Note that for Def-MARL, only the results with `z_min` and `z_max` are saved.
- `--full-eval-interval`: Full evaluation interval where we run the root finding algorithm for Def-MARL, default 10.
- `--eval-epi`: Number of episodes for evaluation, default 1.
- `--save-interval`: Save interval, default 100.
- `--seed`: Random seed, default 0.
- `--steps`: Number of training steps, default 100000.
- `--name`: Name of the experiment, default None.
- `--debug`: Debug mode, in which the logs will not be saved, default False.
- `--gnn-layers`: Number of layers in the actor GNN and the $V^l$ GNN, default 2.
- `--Vh-gnn-layers`: Number of layers in the $V^h$ GNN, default 1.
- `--lr-actor`: Learning rate of the actor, default 3e-4. **Consider changing to 1e-5 for 1 agent.**
- `--lr-critic`: Learning rate of $V^l$ and $V^h$, default 1e-3. **Consider changing to 3e-4 for 1 agent.**
- `--rnn-layers`: Number of layers in the RNN, default 1.
- `--use-lstm`: Use LSTM, default False (use GRU).
- `--rnn-step`: Number of RNN steps in a chunk, default 16.

### Test

To test the learned model, use:

```bash
python test.py --path <path-to-log>
```

This should report the reward, min/max reward, cost, min/max cost, and the safety rate of the learned model. Also, it will generate videos of the learned model in `<path-to-log>/videos`. Use the following flags to customize the test:

#### Required Flags
`--path`: Path to the log.

#### Optional Flags
- `--epi`: Number of episodes for testing, default 5.
- `--no-video`: Do not generate videos, default False.
- `--step`: If this is given, evaluate the model at the given step, default None (evaluate the last model).
- `-n`: Number of agents, default as the same as training.
- `--obs`: Number of obstacles, default as the same as training.
- `--env`: Environment, default as the same as training.
- `--full-observation`: Use full observation, default False.
- `--cpu`: Use CPU only, default False.
- `--max-step`: Maximum number of steps for each episode, default None.
- `--stochastic`: Use stochastic policy, default False.
- `--log`: Log the results, default False.
- `--seed`: Random seed, default 1234.
- `--debug`: Debug mode.
- `--dpi`: DPI of the video, default 100.
- `-z`: [For Def-MARL] Evaluate different z values. Choices are `min` and `max`. If not given, by default evaluate the optimal z value. 
- `--area-size`: Side length of the environment, default None.


## Citation

```
@inproceedings{zhang2025defmarl,
      title={Solving Multi-Agent Safe Optimal Control with Distributed Epigraph Form {MARL}},
      author={Zhang, Songyuan and So, Oswin and Black, Mitchell and Serlin, Zachary and Fan, Chuchu},
      booktitle={Proceedings of Robotics: Science and Systems},
      year={2025},
}
```
