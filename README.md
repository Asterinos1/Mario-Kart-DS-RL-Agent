# Mario Kart DS - RL Agent (DQN/CNN)

## Overview

This project implements an autonomous Reinforcement Learning agent for **Mario Kart DS**. It combines **Convolutional Neural Networks** (CNNs) with **Deep Q-Networks** (DQN). The agent uses a dual-input approach: visual observations (emulated top-screen of the Nintendo DS) and low-level telemetry retrieved directly from the DeSmuME emulator via the `py-desmume` library for reward shaping and richer state information.

Developed as part of the **Autonomous Agents (ΠΛΗ 412, 2025–2026)** course at the **Technical University of Crete**.

---

## Requirements

- Python 3.12.x (developed on Python 3.12.7)
- OS: Tested on Windows 11 but should also work on Linux/MacOS (as long as requirements are compatible)
- Emulator: I used the [DeSmuME](https://desmume.org/) emulator's [py-desmume](https://pypi.org/project/py-desmume/) python library. We only need the latter (it's included in the `requierments.txt`)
- ROM: You must own a legal copy of the Mario Kart DS ROM (USA version).
- GPU is helpful for faster CNN training but not strictly required

---

## Execution workflow

**NOTE: This is a "Quick Start" guide, check out the project's documentation for more details.**

### Setting up the enviroment
1. Clone the repo
```bash
git clone https://github.com/Asterinos1/Mario-Kart-DS-RL-Agent.git
```
2. Create and activate a Python virtual environment (recommended)
3. Install dependencies:
```bash
pip install -r requierments.txt
```
4. Place your Mario Kart DS ROM (USA version) in the `rom/` folder.

### Training the agent
In the current build, the agent is trained on the Figure-8 Circuit (Time Trials) using a specific save state for simplicity. The model learns to navigate the track using three primary inputs: Gas, Gas + Left Steering, and Gas + Right Steering.

To start training:
```bash
python train_sb3_dqn.py
```
You can either let it finish (zzz) or interupt the process at any time with Ctrl+C. You can train new models or resume training previous ones. Models are saved in the /outputs directory.

### Evaluation
Once training is complete or interrupted, you can evaluate the agent's performance by watching it drive or by generating analytical visualizations.

- Watch the Agent Drive:

Run the demonstration script to open a separate window where the agent drives autonomously using the trained model.
```bash
python demo.py
```

- Generate Performance Plots:

Analyze telemetry data (spatial heatmaps and action distributions) and training metrics (learning curves and efficiency).

```bash
python plot_generator.py
```
```bash
python tf_event_parser.py
```
Viewing Results: To see the graphs and plots, navigate to: outputs/{run_id}/plots/


## License

This project is licensed under the MIT License - see the LICENSE file for details.