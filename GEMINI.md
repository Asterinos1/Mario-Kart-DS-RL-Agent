# Mario Kart DS RL Agent — Antigravity Project Rules

## Project Overview

Reinforcement Learning agent for **Mario Kart DS** (USA ROM) using DQN + CNN.
Emulated via `py-desmume` (DeSmuME Python bindings). Trains on the **Figure-8 Circuit (Time Trials)** track.

**Academic context:** Autonomous Agents course (ΠΛΗ 412, 2025–2026), Technical University of Crete.

---

## Tech Stack

| Layer | Library / Tool |
|---|---|
| RL Framework | `stable_baselines3` (DQN) |
| Neural Net Backend | `PyTorch` (CNN policy) |
| Environment | `gymnasium` (custom `MKDSEnv`) |
| Emulator | `py-desmume` / DeSmuME |
| Vision | `opencv-python` (frame preprocessing) |
| Analysis | `matplotlib`, `seaborn`, `tensorflow` (TensorBoard) |
| Python | 3.12.x |

---

## Repository Layout

```
Mario-Kart-DS-RL-Agent/
├── env/
│   └── mkds_gym_env.py       # MKDSEnv — the core Gymnasium environment
├── src/
│   └── utils/
│       ├── config.py         # All hyperparams, RAM addresses, paths
│       ├── callbacks.py      # SB3 callbacks (logging, saving)
│       └── ram_vars_testing.py  # Standalone RAM address discovery / testing
├── analysis/
│   ├── plot_generator.py     # Spatial heatmaps & action distribution plots
│   └── tf_event_parser.py    # Parse TensorBoard events -> CSV/plots
├── train_sb3_dqn.py          # Main training entry-point (SB3 DQN)
├── demo.py                   # Evaluate / watch agent drive
├── requirements.txt          # Pinned Python dependencies
├── mkds_boot.dst             # DeSmuME save state (race start)
├── rom/                      # Put your Mario Kart DS ROM here (git-ignored)
├── outputs/                  # Trained models & plots (git-ignored)
└── logs/                     # TensorBoard logs (git-ignored)
```

> **Graphify graph** (if built): `graphify-out/` -- git-ignored, local only.

---

## Key Design Decisions

### Observation Space
- Top-screen only (192 x 256 px crop from the combined 384 x 256 DS buffer)
- Converted to **grayscale**, resized to `config.STATE_H x config.STATE_W`
- Shape: `(H, W, 1)` -- compatible with SB3's `CnnPolicy`

### Action Space (Discrete, 3 actions)
| Index | Keys | Description |
|---|---|---|
| 0 | A | Accelerate (straight) |
| 1 | A + LEFT | Accelerate + steer left |
| 2 | A + RIGHT | Accelerate + steer right |

### Reward Shaping
- **+speed*2** per step (forward momentum)
- **+15** per checkpoint crossed
- **+100** on race finish (lap > 3)
- **-50** for driving backward
- **-30** for collision/wall scrape (sudden speed drop)
- **-20** for stuck (< 50-unit displacement over ~80 steps)
- **-15** for timeout (no checkpoint in time window)
- **x0.5** multiplier when off-road (`offroad < 0.9`)

### RAM Telemetry (NDS Memory)
All addresses live in `src/utils/config.py`. Key ones:
- `ADDR_BASE_POINTER` -> base struct for physics (speed, angle, position, offroad)
- `ADDR_RACE_INFO_POINTER` -> race state (checkpoint index, lap count)
- `ADDR_TIMER_POINTER` -> 32-bit internal race timer (60 ticks/sec)
- Offsets: `OFFSET_SPEED`, `OFFSET_ANGLE`, `OFFSET_CHECKPOINT`, `OFFSET_LAP`, `OFFSET_OFFROAD`

---

## Common Workflows

### Training
```bash
python train_sb3_dqn.py
```
- Saves model checkpoints to `outputs/` (`.zip` format, git-ignored)
- TensorBoard logs to `logs/DQN_*/` (git-ignored)

### Demo / Evaluation
```bash
python demo.py
```

### Analysis
```bash
python analysis/plot_generator.py   # Heatmaps, action distribution
python analysis/tf_event_parser.py  # Learning curves from TF events
```

---

## Agent Guidelines

- **Do not push** anything in `.gemini/`, `graphify-out/`, or any `outputs/` / `logs/` directories to GitHub -- these are in `.gitignore`.
- **ROM files** (`rom/*.nds`) are proprietary and must never be committed.
- When modifying the reward function, update both `env/mkds_gym_env.py` **and** the documentation table above.
- When changing RAM addresses or offsets, update `src/utils/config.py` and note changes in this file.
- The `mkds_boot.dst` save state file should always be kept in sync with any changes to the boot/reset flow in `MKDSEnv.reset()`.
- Prefer running `graphify` on this directory to get a knowledge graph before doing large refactors -- it surfaces cross-file dependencies quickly.

---

## Graphify Usage

If `graphify-out/graph.json` exists, treat all architecture questions as graphify queries first:

```
/graphify query "How does the reward function work?"
/graphify query "What RAM addresses are used for telemetry?"
/graphify path "MKDSEnv" "config"
```

To build/rebuild the graph:
```
/graphify .
```

The graph output lives in `graphify-out/` which is git-ignored.
