# Soft-Actor-Critic-RL

## Overview
This repository contains the Soft-Actor-Critic implementation the Laser-Hockey environment.  
It includes training and evaluation scripts.

## Project Structure
```text
.
├── submodules/           # Dependencies
│   ├── hockey_env/       # Laser-Hockey Environment
├── results/              # Trained agents
│   ├── agent.pth         # Laser-Hockey trained agent
├── container.def
├── agent.py              # SAC Agent implementation
├── feedforward.py        # Neural networks architectures
├── memory.py             # Replay Buffer
├── obs_scaling.py        # Running Normalizer for observations 
├── SAC_train.py          # Main training script
├── eval_SAC_hockey.py    # Evaluating SAC visually
└── README.md
```

## Installation
1. Clone the repository:
    ```bash
    git clone --recursive git@github.com:Karim-Wahba/SAC-Laser-Hockey.git
    cd SAC-Laser-Hockey
    ```
2. Build singularity container:
    ```bash
    singularity build --fakeroot container.sif container.def
    ```

## Usage

### Train
```bash
singularity run container.sif python3 -u SAC_train.py --debug --alpha 0.2 --finetune --selfplay

```

### Evaluate
```bash
singularity run container.sif python3 -u eval_SAC_hockey.py --checkpoint results/agent.pth
```

## Results Summary
| Algorithm | Environment | Avg Return |
|----------|-------------|------------|
| SAC      | Laser-Hockey         | 9.22         |

