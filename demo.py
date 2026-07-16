"""Demonstration entry-point for the Mario Kart DS DQN agent.

Loads a previously trained DQN model from the ``outputs/`` directory and runs
it in a live, visualised environment so the agent's learned driving behaviour
can be observed.  The emulator window is opened with ``visualize=True`` so the
SDL render surface is visible to the user.

Key design choices:
  - Uses :class:`~stable_baselines3.common.vec_env.DummyVecEnv` (single
    environment, no subprocess overhead) because demo throughput is limited by
    rendering speed rather than CPU parallelism.
  - Sets ``deterministic=False`` during ``model.predict()`` to introduce slight
    stochasticity, which produces visually smoother and more natural-looking
    driving than the fully greedy policy.
  - Tracks per-episode cumulative reward and prints it at episode boundaries
    for quick human evaluation of model quality.

Typical usage::

    python demo.py
"""

import os
import glob
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from env.mkds_gym_env import MKDSEnv
from src.utils import config


def select_model():
    """Scans for saved ``.zip`` models and lets the user choose one to run.

    Performs a recursive glob under ``outputs/`` to discover every ``.zip``
    model file regardless of nesting depth, presents a numbered menu, and
    returns the chosen path stripped of its ``.zip`` extension.

    Note:
        SB3's ``DQN.load()`` expects the path *without* the ``.zip`` suffix;
        the extension is appended internally by the framework.

    Returns:
        str | None: The selected model file path with the ``.zip`` extension
            removed (ready to pass directly to ``DQN.load()``), or ``None``
            if no models are found or the user provides invalid input.
    """
    # Search in outputs root and all sub-folders recursively.
    search_pattern = os.path.join("outputs", "**", "*.zip")
    model_files = glob.glob(search_pattern, recursive=True)

    if not model_files:
        print("No .zip models found in the /outputs directory.")
        return None

    print("\n--- Available Models ---")
    for i, file_path in enumerate(model_files, 1):
        # Show only the path relative to outputs/ to keep the menu readable.
        display_name = os.path.relpath(file_path, "outputs")
        print(f"{i}) {display_name}")

    while True:
        try:
            choice = int(input(f"\nSelect a model to run (1-{len(model_files)}): "))
            if 1 <= choice <= len(model_files):
                selected_path = model_files[choice - 1]
                # Strip .zip so SB3 can append it internally per its convention.
                return os.path.splitext(selected_path)[0]
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a valid number.")


def run_demo():
    """Load a trained DQN model and run it in a live visualised environment.

    Workflow:
    1. Prompts the user to select a model via :func:`select_model`.
    2. Wraps a single :class:`~env.mkds_gym_env.MKDSEnv` (``visualize=True``)
       in a :class:`~stable_baselines3.common.vec_env.DummyVecEnv` to satisfy
       the SB3 vectorised-environment interface without spawning a subprocess.
    3. Applies :class:`~stable_baselines3.common.vec_env.VecFrameStack` to
       match the observation format the model was trained on.
    4. Runs an infinite predict-step loop, printing cumulative episode rewards
       at each episode boundary.
    5. Calls ``base_env.emu.destroy()`` in a ``finally`` block to cleanly shut
       down the emulator regardless of how the loop exits.

    Note:
        ``deterministic=False`` is used intentionally: stochastic action
        selection (via the epsilon-greedy exploration policy) tends to produce
        smoother, more watchable driving than a fully greedy policy, which can
        get stuck repeating a single optimal action sequence.
    """
    model_path = select_model()
    if not model_path:
        return

    print(f"\nInitializing Mario Kart DS Environment with model: {os.path.basename(model_path)}...")

    # Instantiate the base environment with the SDL window enabled so the user
    # can watch the agent drive in real time.
    base_env = MKDSEnv(visualize=True)

    # DummyVecEnv wraps a single environment in the VecEnv interface without
    # creating a subprocess -- ideal for demo/inference where parallelism is
    # unnecessary and would only add IPC overhead.
    env = DummyVecEnv([lambda: base_env])

    # Stack frames to match the observation shape the model was trained on.
    env = VecFrameStack(env, n_stack=config.STACK_SIZE, channels_order='last')

    try:
        model = DQN.load(model_path, env=env, device="cuda")
        print("DQN Model loaded successfully.")
    except Exception as e:
        print(f"Error: Model didn't load properly. {e}")
        # Destroy the emulator before returning to avoid orphaned processes.
        base_env.emu.destroy()
        return

    obs = env.reset()
    episode_count = 1
    current_episode_reward = 0  # Accumulates reward over the current episode.

    print(f"\n--- Starting Episode {episode_count} ---")
    print("Focus the SDL Window to see the agent drive.")
    print("Press Ctrl+C in this terminal to stop.")

    try:
        while True:
            # deterministic=False: allow the policy to sample non-greedy actions.
            # This adds slight variance that produces more natural driving behaviour
            # compared to always picking the single highest-Q action.
            action, _states = model.predict(obs, deterministic=False)

            # Advance the environment by one timestep with the chosen action.
            obs, rewards, dones, infos = env.step(action)

            # rewards is a length-1 array (one env); index [0] gives the scalar.
            current_episode_reward += rewards[0]

            if dones[0]:
                # Episode boundary: print summary and reset the episode counter.
                print(f"Episode {episode_count} Finished | Reward: {current_episode_reward:.2f}")
                episode_count += 1
                current_episode_reward = 0  # Reset accumulator for the new episode.
                print(f"\n--- Starting Episode {episode_count} ---")

    except KeyboardInterrupt:
        print("\nDemonstration stopped by user.")
    finally:
        # Always destroy the emulator to release the SDL window and any
        # underlying DeSmuME resources, even if an exception occurred.
        base_env.emu.destroy()
        print("Emulator closed.")


if __name__ == "__main__":
    run_demo()
