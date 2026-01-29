# Mario Kart DS - RL Agent (DQN/CNN)

## Overview

This project implements an autonomous Reinforcement Learning agent for **Mario Kart DS**. It combines **Convolutional Neural Networks** (CNNs) with **Deep Q-Networks** (DQN) and optionally uses **Stable-Baselines3** (SB3) for experimentation. The agent uses a dual-input approach: visual observations (top-screen captures) and low-level telemetry retrieved directly from the DeSmuME emulator via the `py-desmume` library for reward shaping and richer state information.

Developed as part of the **Autonomous Agents (ΠΛΗ 412, 2025–2026)** course at the **Technical University of Crete**.

---

## Quick start

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
5. Run either of the training scripts. Once finished, run the corresponding demo script to preview the final model

---

## Requirements

- Python 3.12.x (developed on Python 3.12.7)
- OS: Test on Windows 11 but should also on Linux/MacOS
- Emulator: I used the [DeSmuME](https://desmume.org/) emulator's [py-desmume](https://pypi.org/project/py-desmume/) python library. We only need the latter (it's included in the `requierments.txt`)
- ROM: You must own a legal copy of the Mario Kart DS ROM (USA version).
- GPU is helpful for faster CNN training but not strictly required
