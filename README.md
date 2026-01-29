# Mario Kart DS RL Agent (DQN / CNN)

This repository contains a Reinforcement Learning agent designed to play **Mario Kart DS (USA Version)** autonomously. The project utilizes **DQN**, **CNNs**, **Stable-Baselines3 (SB3)**, and **Gymnasium** to teach an agent to play **Mario Kart DS (USA)**. The agent combines **visual input** (top screen capture) with **internal emulator state** (RAM access) via **DeSmuME**.

Developed as part of the course **Autonomous Agents ΠΛΗ 412 (2025-2026)** at the **Technical University of Crete**.


---

## Project Structure

```text
root/
├── .gitignore
├── .vscode/
├── env/
│   ├── mkds_custom_env.py        # Custom Gymnasium env (DQN training)
│   └── mkds_gym_env.py           # Gymnasium env (SB3 training)
├── envs/
│   └── mkds_env/                 # Python virtual environment
├── outputs/                      # Checkpoints + final trained model (.zip)
├── logs/                         # Training logs (DQN_1, DQN_2, ...)
├── rom/                          # Placeholder for ROM (PUT ROM HERE, MUST BE USA VERSION)
├── src/                          
│   ├── agents/
│   │   └── dqn_agent.py          # Custom DQN implementation
│   └── utils/
│       ├── config.py             # Global configs & hyperparameters
│       ├── visualization.py      # Training & gameplay visualizations
│       ├── wrappers.py           # Observation/action wrappers
│       └── ram_vars_testing.py   # RAM access validation script
├── mkds_boost.dst                # Save-state (race start position)
├── requierments.txt              # Python dependencies
├── train_tf_dqn.py               # Secondary training entry (TF DQN)
├── train_sb3_ppo.py              # Primary training entry (SB3 PPO)
├── demo_sb3.py                   # Run trained model (inference/demo)
├── README.md
└── emu_documentation.txt         # py-desmume API notes & summary
