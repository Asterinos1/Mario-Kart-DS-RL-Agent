# Mario Kart DS RL Agent — Improvement Roadmap

---

## 1. Open Bugs (Not Yet Fixed)

These are the remaining bugs from the assessment. Highest ROI per line of code.

| # | File | Bug | Fix |
|---|---|---|---|
| 🔴 | `mkds_gym_env.py:135` | **Timeout is 10 ticks (0.17s)** — episodes terminate almost instantly | Change `> 10` → `> 300` |
| 🔴 | `mkds_gym_env.py:126` | **Lap rollover triggers false "backward" termination** every new lap | Guard with `and lap == self.prev_lap` or handle the rollover case |
| 🟡 | `train_sb3_dqn.py:41,48` | `device="cuda"` hardcoded — crashes on CPU-only machines | Change to `device="auto"` in both train and demo |
| 🟡 | `mkds_gym_env.py` | `close()` method not implemented — emulator resource leak | Add `def close(self): self.emu.destroy()` |
| 🟡 | `train_sb3_dqn.py:64` | `total_timesteps=1_000_000` hardcoded | Move to `config.TOTAL_TIMESTEPS` |

---

## 2. RL Algorithm

### 2a. Stabilize DQN Training
**Effort: Low — Impact: High**

SB3's DQN already uses Double DQN by default, but these parameters are not tuned for a visual racing task:

- **Replay buffer is too small.** 50k transitions for a 4-frame stacked visual env is undersized. Atari benchmarks use 1M. Even 200k would help significantly with coverage of diverse states.
- **Batch size of 128 is fine**, but `learning_starts` (when gradient updates begin) defaults to 50k in SB3 — this means the first 50k steps produce no learning at all. With 4 envs, that's 12.5k "real" steps. Consider dropping it to 10k.
- **Target network update.** SB3's DQN defaults `target_update_interval=10000`. For a racing task with fast-changing dynamics, a shorter interval (e.g. 5000) may improve stability.
- **`VecNormalize` wrapper.** Rewards range from −50 to +100+. Without normalization, Q-value estimates can diverge. Wrapping with `VecNormalize(norm_reward=True)` is a one-liner that often meaningfully stabilizes training.

### 2b. Upgrade to Rainbow DQN Components
**Effort: Medium — Impact: High**

SB3 DQN supports some but not all Rainbow components. The two most impactful for this task:

- **Prioritized Experience Replay (PER)** — replays transitions with high TD-error more often. Critical for sparse reward tasks where "crash" transitions are informative. Available via `sb3-contrib`.
- **Dueling network** — separates value and advantage streams. Helps in states where the action barely matters (straight sections). Can be implemented via a custom policy.
- **n-step returns** — compute multi-step TD targets instead of single-step. Reduces variance, especially helpful with the delayed checkpoint reward (+15 only comes when a checkpoint is crossed).

### 2c. Alternative Algorithm: PPO
**Effort: High — Impact: Potentially High**

PPO (Proximal Policy Optimization) is generally more sample-efficient than DQN on continuous control tasks and has better exploration properties. The discrete 3-action space is compatible. The tradeoff: PPO requires on-policy data, meaning no replay buffer — fewer total steps before convergence but each step is more expensive to compute.

---

## 3. Observation Space

### 3a. Add RAM Telemetry as Auxiliary Input
**Effort: Medium — Impact: High**

Currently the agent is purely visual. But `config.py` already has 25+ verified RAM offsets — speed, checkpoint progress, surface type, drift charge, boost state, etc. Adding even a small telemetry vector as a second input channel could dramatically reduce the number of steps needed to learn basic driving:

```python
# Example: telemetry vector alongside the visual obs
obs_space = spaces.Dict({
    "screen": spaces.Box(0, 255, (84, 84, 4), dtype=np.uint8),
    "telemetry": spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32),
    # [normalized_speed, offroad_modifier, checkpoint_fraction, lap_fraction]
})
```

SB3 supports `Dict` obs with `MultiInputPolicy`. This is a significant architecture change but the data is already being read in `_read_ram()` — it's just not being passed to the policy.

### 3b. Use the Mini-Turbo Charge Offset
**Effort: Low — Impact: Medium**

`OFFSET_MT_CHARGE` is already discovered and unused. Mini-turbo (drift boost) is a core Mario Kart mechanic. Including `mt_charge` in the telemetry vector and adding a small reward for executing a successful boost (`mt_boost > 0`) would encourage the agent to learn drift behavior.

### 3c. Larger / Color Observation
**Effort: Low — Impact: Low-Medium**

The current 84×84 grayscale is standard Atari sizing. For a driving task where track color cues matter (green grass vs. grey road), **color input** (84×84×3 per frame) could help the agent learn the off-road distinction visually rather than relying purely on the RAM `offroad` modifier. The compute cost increases ~3×.

---

## 4. Action Space

### 4a. Re-enable the 6-Action Drift Space
**Effort: Low — Impact: High**

The 6-action space is already fully implemented and commented out in both `mkds_gym_env.py` and `config.py`:

```python
# 3: Drift+Straight, 4: Drift+Left, 5: Drift+Right
```

Drift (R button) is the primary skill gap between a naive agent and a competent one. On Figure-8 Circuit specifically, the two tight corners are where lap time is won or lost. Enabling drift gives the agent the tools to actually take optimal lines. The bigger action space will slow initial learning but is worth it past ~200k steps.

### 4b. Add Braking (B Button)
**Effort: Low — Impact: Low**

A brake action (B alone, or B + steering) would allow the agent to scrub speed before corners. Likely low impact on Figure-8 (a low-difficulty track) but necessary for any transfer to harder tracks.

---

## 5. Reward Function

### 5a. Fix the Two Open Reward Bugs
See Section 1 — timeout and lap rollover. These are the most urgent.

### 5b. Anti-Oscillation Penalty
**Effort: Low — Impact: Medium**

A common failure mode for DQN on racing games is rapid left-right oscillation on straightaways. This emerges because the agent learns left and right are "about equal" on a straight. A small penalty for switching steering direction on consecutive steps discourages this:

```python
if action != 0 and action == 3 - self.prev_action:  # flipped steering
    reward -= 0.5
```

### 5c. Lap Time Reward
**Effort: Low — Impact: Medium**

Currently there's no incentive to complete laps *quickly* — the +100 finish bonus is the same regardless of time. Adding a time-based bonus (e.g. `+50` if lap time is under a threshold, read from `ADDR_TIMER_POINTER`) would encourage the agent to optimize for speed, not just completion.

### 5d. Curriculum Learning via Save States
**Effort: Medium — Impact: High**

Currently the agent always starts at the same race-start position. One powerful technique: create multiple save states at different points along the track (mid-corner, post-corner, etc.) and randomly sample the start position each episode. This exposes the agent to diverse states much earlier in training, dramatically improving sample efficiency.

---

## 6. Training Infrastructure

### 6a. Argument Parsing (argparse / Hydra)
**Effort: Low — Impact: Medium**

Right now to change any hyperparameter you edit `config.py`. Adding `argparse` to `train_sb3_dqn.py` allows overriding at launch:

```bash
python train_sb3_dqn.py --total-timesteps 2000000 --batch-size 256 --n-envs 8
```

**Hydra** (Meta's config framework) goes further — it enables YAML-based experiment configs with automatic output directory management, making it trivial to reproduce or compare runs.

### 6b. Experiment Tracking: Weights & Biases
**Effort: Low — Impact: Medium**

SB3 has a built-in W&B callback (`WandbCallback`). This gives you a hosted dashboard with automatic hyperparameter logging, run comparison, and shareable experiment links — useful for an academic submission. TensorBoard is local-only; W&B is accessible from any browser.

### 6c. GitHub Actions CI
**Effort: Medium — Impact: Medium**

A basic CI pipeline that runs on every push:
- **Lint** with `ruff` or `flake8`
- **Type check** with `mypy`
- **Import smoke test** (verifies no import-time side effects like the fixed config.py bug)
- **Unit tests** for reward logic (mock the RAM reads, verify watchdog triggers)

This would catch regressions immediately rather than discovering them mid-training run.

### 6d. Pre-commit Hooks
**Effort: Low — Impact: Low**

`.pre-commit-config.yaml` with `black` (formatter) + `ruff` (linter) ensures consistent code style without thinking about it. One-time setup, zero ongoing effort.

---

## 7. Code Quality

### 7a. Type Hints Throughout
**Effort: Low — Impact: Low-Medium**

`mkds_gym_env.py` and `callbacks.py` have no type annotations. Adding them catches bugs at development time (e.g. `_read_ram()` returns a tuple — its callers assume a specific order) and makes the codebase readable to new contributors.

### 7b. Replace `print` with `logging`
**Effort: Low — Impact: Low**

All feedback currently goes to stdout via `print`. Replacing with Python's `logging` module allows:
- Log level filtering (DEBUG vs INFO vs WARNING)
- Redirecting logs to file during long training runs
- Suppressing emulator noise while keeping agent info

### 7c. Config Validation with Pydantic
**Effort: Medium — Impact: Low-Medium**

Replacing `config.py` constants with a `pydantic.BaseModel` or `dataclass` gives:
- Type-checked fields at startup
- Range validation (e.g. `LEARNING_RATE` must be > 0)
- Serialization to JSON for experiment logging

### 7d. Unit Tests for Reward Logic
**Effort: Medium — Impact: Medium**

The watchdog system in `step()` is complex enough to warrant tests. Mock the emulator with a fake RAM state and verify:
- Backward driving correctly triggers at `cp < prev_cp` with same lap
- Lap rollover does NOT trigger backward (after the lap bug is fixed)
- Timeout triggers exactly at 300 ticks
- Collision fires only when `speed < 5 and speed_drop > 3`

---

## 8. Evaluation & Analysis

### 8a. Proper Evaluation Loop in `demo.py`
**Effort: Low — Impact: Medium**

Currently `demo.py` runs indefinitely. A proper evaluation mode would:
- Run N fixed episodes
- Report mean ± std reward, completion rate, mean lap time
- Save a summary JSON to `outputs/<run_id>/eval_results.json`

This gives a reproducible metric to compare checkpoints scientifically.

### 8b. Video Recording
**Effort: Low — Impact: Medium**

SB3's `VecVideoRecorder` wrapper can save `.mp4` recordings of evaluation episodes automatically. Much easier to share/document than a GIF and captures full episodes.

### 8c. Fixed-Point Scaling on Heatmap
**Effort: Trivial — Impact: Low**

`plot_generator.py` plots raw NDS fixed-point coordinates. Dividing `pos_x` and `pos_z` by `4096.0` would make the heatmap axes represent real in-game meters, making it possible to overlay the track layout.

---

## 9. Advanced / Research-Level

These are higher-effort ideas more appropriate for a research extension than the current scope.

| Idea | Description |
|---|---|
| **Imitation learning warmup** | Record a few human laps via `ram_vars_testing.py` (already has keyboard control), convert to transitions, and pre-train the policy with Behavioural Cloning before RL fine-tuning. Dramatically reduces early random exploration. |
| **Transfer to other tracks** | Figure-8 Circuit is trivially simple. Training on Yoshi Falls, Cheep Cheep Beach etc. would be a much more interesting result. The ROM already has all tracks; creating new save states is the main effort. |
| **Model-based RL** | Learn a world model (predict next frame + RAM state from current frame + action) and plan using it. Dramatically more sample-efficient but very complex to implement correctly. |
| **Multi-task learning** | Train a single policy across multiple tracks simultaneously. Requires curriculum and per-track save states. |
| **Opponent awareness** | Grand Prix mode adds CPU opponents. Adding opponent positions to the observation (via RAM offsets for other karts) would expose the agent to overtaking and defensive driving. |

---

## Priority Order (Suggested)

```
1. Fix timeout bug (10 → 300 ticks)             — 1 line, game-changing
2. Fix lap rollover false-backward               — 5 lines, fixes every lap transition
3. Increase replay buffer (50k → 200k+)          — 1 config change
4. Add VecNormalize                              — 2 lines in train_sb3_dqn.py
5. Re-enable 6-action drift space               — uncomment + test
6. Anti-oscillation penalty                      — 3 lines in step()
7. Add RAM telemetry to observation              — medium effort, high upside
8. Argparse / Hydra for experiment config        — quality of life
9. GitHub Actions CI                             — catch regressions
10. Evaluation loop + video recording            — academic credibility
```
