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

    python demo.py --model DQN_0716_1200 --deterministic
"""

import os
import glob
import argparse
import logging
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from env.mkds_gym_env import MKDSEnv
from src.utils import config, setup_logging

logger = logging.getLogger(__name__)



def parse_args():
    """Parses command-line arguments for evaluation and demonstration options."""
    parser = argparse.ArgumentParser(
        description="Run a trained Mario Kart DS DQN agent in demonstration mode."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to a model checkpoint (.zip) or a run ID in outputs/ to demo. "
             "If not specified, the interactive selection menu will be displayed.",
    )
    parser.add_argument(
        "--stack-size",
        type=int,
        default=config.STACK_SIZE,
        help=f"Number of consecutive frames stacked per observation (default: {config.STACK_SIZE})",
    )
    parser.add_argument(
        "--action-space",
        type=int,
        default=config.ACTION_SPACE,
        choices=[3, 6],
        help=f"Number of discrete actions: 3 (basic) or 6 (with drift) (default: {config.ACTION_SPACE})",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Run model predictions deterministically (default is stochastic to produce smoother driving).",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Run the environment headless (without the SDL visual window).",
    )
    return parser.parse_args()


def resolve_demo_model_path(model_arg):
    """Resolves a model path or a run ID to a model file path without .zip extension.

    Args:
        model_arg (str): A filename, file path, or run ID.

    Returns:
        str: Absolute-style path to the model file ready to load (without .zip extension).

    Raises:
        FileNotFoundError: If the specified run or model could not be found.
    """
    # 1. Check if it's a direct path to a model file (with or without .zip extension)
    clean_arg = model_arg
    if clean_arg.endswith(".zip"):
        clean_arg = clean_arg[:-4]
    
    zip_path = clean_arg + ".zip"
    if os.path.isfile(zip_path):
        return os.path.abspath(clean_arg)

    # 2. Check if it's a run ID or directory name
    run_id = os.path.basename(os.path.normpath(model_arg))
    run_dir = os.path.normpath(os.path.join("outputs", run_id))
    
    if os.path.isdir(run_dir):
        model_files = glob.glob(os.path.join(run_dir, "models", "*.zip"))
        if model_files:
            latest_model = max(model_files, key=os.path.getmtime)
            # Strip .zip extension
            return os.path.splitext(os.path.abspath(latest_model))[0]
        else:
            raise FileNotFoundError(f"No model checkpoints (.zip) found in {run_dir}/models/")
    else:
        raise FileNotFoundError(
            f"Could not resolve model argument '{model_arg}'. "
            f"It is not an existing model file, and no run folder found at outputs/{run_id}"
        )


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


def run_demo(args=None):
    """Load a trained DQN model and run it in a live visualised environment.

    Workflow:
    1. Prompts the user to select a model via :func:`select_model` (or uses the CLI argument).
    2. Wraps a single :class:`~env.mkds_gym_env.MKDSEnv` (``visualize=True`` unless headless)
       in a :class:`~stable_baselines3.common.vec_env.DummyVecEnv` to satisfy
       the SB3 vectorised-environment interface without spawning a subprocess.
    3. Applies :class:`~stable_baselines3.common.vec_env.VecFrameStack` to
       match the observation format the model was trained on.
    4. Runs an infinite predict-step loop, printing cumulative episode rewards
       at each episode boundary.
    5. Calls ``base_env.emu.destroy()`` in a ``finally`` block to cleanly shut
       down the emulator regardless of how the loop exits.
    """
    if args is None:
        args = parse_args()

    # Initialize console logging
    setup_logging()

    # Override config values
    config.STACK_SIZE = args.stack_size
    config.ACTION_SPACE = args.action_space

    if args.model:
        try:
            model_path = resolve_demo_model_path(args.model)
        except FileNotFoundError as e:
            logger.error(f"Error loading model: {e}")
            return
    else:
        model_path = select_model()

    if not model_path:
        return

    logger.info(f"Initializing Mario Kart DS Environment with model: {os.path.basename(model_path)}...")

    # Instantiate the base environment. We support toggling visualization.
    visualize = not args.no_visualize
    base_env = MKDSEnv(visualize=visualize)

    # DummyVecEnv wraps a single environment in the VecEnv interface without
    # creating a subprocess -- ideal for demo/inference where parallelism is
    # unnecessary and would only add IPC overhead.
    env = DummyVecEnv([lambda: base_env])

    # Stack frames to match the observation shape the model was trained on.
    env = VecFrameStack(env, n_stack=config.STACK_SIZE, channels_order='last')

    try:
        model = DQN.load(model_path, env=env, device="auto")
        logger.info("DQN Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error: Model didn't load properly. {e}")
        # Destroy the emulator before returning to avoid orphaned processes.
        base_env.emu.destroy()
        return

    obs = env.reset()
    episode_count = 1
    current_episode_reward = 0  # Accumulates reward over the current episode.

    logger.info(f"--- Starting Episode {episode_count} ---")
    if visualize:
        logger.info("Focus the SDL Window to see the agent drive.")
    logger.info("Press Ctrl+C in this terminal to stop.")

    try:
        while True:
            # deterministic: allow user to force greedy actions if requested.
            action, _states = model.predict(obs, deterministic=args.deterministic)

            # Advance the environment by one timestep with the chosen action.
            obs, rewards, dones, infos = env.step(action)

            # rewards is a length-1 array (one env); index [0] gives the scalar.
            current_episode_reward += rewards[0]

            if dones[0]:
                # Episode boundary: print summary and reset the episode counter.
                logger.info(f"Episode {episode_count} Finished | Reward: {current_episode_reward:.2f}")
                episode_count += 1
                current_episode_reward = 0  # Reset accumulator for the new episode.
                logger.info(f"--- Starting Episode {episode_count} ---")

    except KeyboardInterrupt:
        logger.info("Demonstration stopped by user.")
    finally:
        # Always destroy the emulator to release the SDL window and any
        # underlying DeSmuME resources, even if an exception occurred.
        base_env.emu.destroy()
        logger.info("Emulator closed.")


if __name__ == "__main__":
    run_demo()
