"""Training entry-point for the Mario Kart DS DQN agent using Stable-Baselines3.

This script orchestrates the full training loop:
  - Optionally resumes from a previously saved checkpoint (model + replay buffer).
  - Spins up parallel emulator subprocesses via SubprocVecEnv for data collection.
  - Stacks consecutive frames with VecFrameStack to give the agent temporal context.
  - Periodically saves model checkpoints and the replay buffer so training can be
    resumed at any point without losing collected experience.
  - Intercepts Ctrl+C and performs a guaranteed "safety save" before exit.

Typical usage::

    python train_sb3_dqn.py --total-timesteps 2000000 --batch-size 256 --n-envs 8
"""

import os
import glob
import argparse
import logging
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from env.mkds_gym_env import MKDSEnv
from src.utils.callbacks import MKDSMetricsCallback
from src.utils import config, setup_logging

logger = logging.getLogger(__name__)



def parse_args():
    """Parses command-line arguments for training hyper-parameters and options."""
    parser = argparse.ArgumentParser(
        description="Train a Mario Kart DS DQN agent using Stable-Baselines3."
    )
    
    # Run configuration / resume options
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a model checkpoint (.zip) or a run ID in outputs/ to resume. "
             "If a run ID is provided, the latest checkpoint within that run will be loaded.",
    )
    group.add_argument(
        "--fresh",
        action="store_true",
        help="Skip the interactive resume menu and start a new training run.",
    )

    # RL Hyperparameters
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=config.TOTAL_TIMESTEPS,
        help=f"Total training timesteps (default: {config.TOTAL_TIMESTEPS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help=f"Minibatch size for each gradient update (default: {config.BATCH_SIZE})",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=config.NUM_OF_INSTANCES,
        help=f"Number of parallel emulator environments (default: {config.NUM_OF_INSTANCES})",
    )
    parser.add_argument(
        "--stack-size",
        type=int,
        default=config.STACK_SIZE,
        help=f"Number of consecutive frames stacked per observation (default: {config.STACK_SIZE})",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=config.GAMMA,
        help=f"Discount factor (default: {config.GAMMA})",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=config.LEARNING_RATE,
        help=f"Learning rate for Adam optimizer (default: {config.LEARNING_RATE})",
    )
    parser.add_argument(
        "--buffer-size",
        "--memory-size",
        type=int,
        default=config.MEMORY_SIZE,
        help=f"Maximum replay buffer capacity (default: {config.MEMORY_SIZE})",
    )
    parser.add_argument(
        "--action-space",
        type=int,
        default=config.ACTION_SPACE,
        choices=[3, 6],
        help=f"Number of discrete actions: 3 (basic) or 6 (with drift) (default: {config.ACTION_SPACE})",
    )
    
    # Directories / Logging
    parser.add_argument(
        "--tb-log-dir",
        type=str,
        default="./logs/",
        help="Directory to write TensorBoard logs (default: ./logs/)",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=10000,
        help="Checkpoint saving frequency in environment steps (default: 10000)",
    )

    return parser.parse_args()


def resolve_resume_path(resume_arg):
    """Resolves a model path or a run ID to an absolute-style model file path.

    Args:
        resume_arg (str): A filename, file path, or run ID.

    Returns:
        tuple[str, str]: A 2-tuple of (run_id, model_path).

    Raises:
        FileNotFoundError: If the specified run or model could not be found.
    """
    # 1. Check if it's a direct path to a .zip model file
    if os.path.isfile(resume_arg) and resume_arg.endswith(".zip"):
        path_parts = os.path.normpath(resume_arg).split(os.sep)
        try:
            # e.g., outputs/DQN_xxxx/models/ckpt.zip -> run_id = DQN_xxxx
            idx = path_parts.index("outputs")
            run_id = path_parts[idx + 1]
        except (ValueError, IndexError):
            run_id = "resumed_run"
        return run_id, os.path.abspath(resume_arg)

    # 2. Check if it's a run ID or directory name
    run_id = os.path.basename(os.path.normpath(resume_arg))
    run_dir = os.path.normpath(os.path.join("outputs", run_id))
    
    if os.path.isdir(run_dir):
        model_files = glob.glob(os.path.join(run_dir, "models", "*.zip"))
        if model_files:
            latest_model = max(model_files, key=os.path.getmtime)
            return run_id, os.path.abspath(latest_model)
        else:
            raise FileNotFoundError(f"No model checkpoints (.zip) found in {run_dir}/models/")
    else:
        raise FileNotFoundError(
            f"Could not resolve resume argument '{resume_arg}'. "
            f"It is not an existing .zip file, and no run folder found at outputs/{run_id}"
        )


def select_resume_option():
    """Scans the outputs directory and presents an interactive resume menu.

    Walks every subdirectory of ``outputs/``, collects the most-recently
    modified ``.zip`` model file from each run's ``models/`` folder, and
    lets the user pick one to resume.  Pressing Enter (empty input) starts
    a fresh run instead.

    Returns:
        tuple[str | None, str | None]: A 2-tuple of ``(run_id, model_path)``
            where *run_id* is the name of the chosen run directory and
            *model_path* is the absolute-style path to the ``.zip`` file.
            Returns ``(None, None)`` when:
            - the ``outputs/`` directory does not exist,
            - no run contains a saved model, or
            - the user presses Enter to start a new run.
    """
    # Nothing to resume if the outputs root has not been created yet.
    if not os.path.exists("outputs"):
        return None, None

    # Collect every top-level subdirectory (one per training run).
    runs = [d for d in os.listdir("outputs") if os.path.isdir(os.path.join("outputs", d))]

    options = []
    for run in runs:
        # Find all checkpoint zips saved inside this run's models folder.
        model_files = glob.glob(f"outputs/{run}/models/*.zip")
        if model_files:
            # Keep only the *latest* checkpoint; mtime is reliable here because
            # SB3 checkpoint filenames embed the step count anyway.
            latest_model = max(model_files, key=os.path.getmtime)
            options.append((run, latest_model))

    # No resumable runs found — fall through to a fresh start.
    if not options:
        return None, None

    print("\n--- Available Models to Resume ---")
    for i, (run_id, path) in enumerate(options):
        print(f"{i}: {run_id} ({os.path.basename(path)})")

    choice = input(f"\nSelect index (Enter for NEW): ")

    # Validate: must be a non-negative integer within the displayed range.
    return options[int(choice)] if choice.isdigit() and int(choice) < len(options) else (None, None)


def train(args=None):
    """Main training loop for the Mario Kart DS DQN agent.

    Performs the following steps in order:

    1. **Resume or fresh start** -- checks command-line options or calls
       :func:`select_resume_option` to decide whether to load an existing
       checkpoint (model + replay buffer) or initialise a brand-new DQN.
    2. **Environment setup** -- creates *N* parallel emulator processes with
       :class:`~stable_baselines3.common.vec_env.SubprocVecEnv` (where *N* is
       ``config.NUM_OF_INSTANCES``), then wraps them in
       :class:`~stable_baselines3.common.vec_env.VecFrameStack` so each
       observation contains ``config.STACK_SIZE`` consecutive frames stacked
       along the channel axis, giving the CNN temporal awareness.
    3. **Training** -- calls ``model.learn()`` for up to TOTAL_TIMESTEPS
       with two callbacks running in parallel:
       - :class:`~src.utils.callbacks.MKDSMetricsCallback` -- logs custom
         game metrics (speed, position, lap) to a CSV inside the run folder.
       - :class:`~stable_baselines3.common.callbacks.CheckpointCallback` --
         saves a model *and* the full replay buffer every save_freq steps so
         off-policy learning can be resumed warm (no cold-start penalty).
    4. **Safety save** -- a ``try/finally`` block guarantees that the current
       model and replay buffer are written to disk even when the user presses
       Ctrl+C mid-training.

    Raises:
        KeyboardInterrupt: Caught internally; triggers the safety save and a
            graceful environment shutdown before re-raising via ``finally``.
    """
    if args is None:
        args = parse_args()

    # Initialize console logging
    setup_logging()

    # Override configuration defaults in the config module so that any imported
    # modules (like MKDSEnv) will dynamically use the CLI argument values.
    config.NUM_OF_INSTANCES = args.n_envs
    config.STACK_SIZE = args.stack_size
    config.TOTAL_TIMESTEPS = args.total_timesteps
    config.MEMORY_SIZE = args.buffer_size
    config.BATCH_SIZE = args.batch_size
    config.GAMMA = args.gamma
    config.LEARNING_RATE = args.learning_rate
    config.ACTION_SPACE = args.action_space

    if args.resume:
        try:
            run_id, model_path = resolve_resume_path(args.resume)
        except FileNotFoundError as e:
            logger.error(f"Error resuming: {e}")
            return
    elif args.fresh:
        run_id, model_path = None, None
    else:
        run_id, model_path = select_resume_option()

    # TensorBoard logs are written to a single shared directory so that
    # multiple runs can be compared side-by-side in one TB session.
    tb_log_path = args.tb_log_dir

    # --- Environment setup ---
    # SubprocVecEnv spawns each MKDSEnv in its own subprocess, enabling true
    # CPU parallelism for data collection (one emulator instance per process).
    # visualize=False disables the SDL render window in worker processes to
    # avoid GPU/display contention and speed up frame generation.
    env = SubprocVecEnv([lambda: MKDSEnv(visualize=False) for _ in range(config.NUM_OF_INSTANCES)])

    # VecFrameStack concatenates the last STACK_SIZE observations along the
    # channel axis (channels_order='last' -> HWC layout expected by SB3's CNN).
    # This turns a single 2-D frame into a short video clip the CNN can use to
    # infer velocity and direction -- critical for a racing game.
    env = VecFrameStack(env, n_stack=config.STACK_SIZE, channels_order='last')

    if model_path:
        # --- Resume an existing run ---
        logger.info(f"--- Resuming: {run_id} ---")

        # Overrides of hyperparameters passed via command line
        custom_objects = {
            "learning_rate": config.LEARNING_RATE,
            "gamma": config.GAMMA,
            "batch_size": config.BATCH_SIZE,
            "buffer_size": config.MEMORY_SIZE,
        }

        # Reload weights and hyper-parameters; bind the resumed model to the
        # freshly created vectorised environment.
        model = DQN.load(model_path, env=env, device="auto", tensorboard_log=tb_log_path, custom_objects=custom_objects)

        # The replay buffer is saved alongside the model checkpoint as a .pkl
        # file.  Loading it lets DQN continue off-policy learning immediately
        # without refilling the buffer from scratch (warm resumption).
        buffer_path = model_path.replace(".zip", "_replay_buffer.pkl")
        if os.path.exists(buffer_path):
            model.load_replay_buffer(buffer_path)
    else:
        # --- Fresh run ---
        # Timestamp-based run ID ensures unique output folders for every run.
        run_id = f"DQN_{datetime.now().strftime('%m%d_%H%M')}"
        logger.info(f"--- Fresh Run: {run_id} ---")

        model = DQN(
            "CnnPolicy",      # Convolutional policy suited for pixel observations.
            env,
            verbose=1,
            device="auto",
            buffer_size=config.MEMORY_SIZE,   # Maximum replay buffer capacity (transitions).
            batch_size=config.BATCH_SIZE,     # Minibatch size for each gradient update.
            learning_rate=config.LEARNING_RATE, # Adam learning rate.
            gamma=config.GAMMA,               # Discount factor.
            tensorboard_log=tb_log_path,
        )

    # Create per-run output directories (safe to call even if they already exist).
    base_path = f"outputs/{run_id}"
    os.makedirs(f"{base_path}/models", exist_ok=True)
    os.makedirs(f"{base_path}/logs", exist_ok=True)

    # Configure logging to write to file as well
    setup_logging(log_file=f"{base_path}/logs/train.log")

    # --- Callbacks ---
    # CallbackList executes both callbacks at every step simultaneously.
    callbacks = CallbackList([
        # Custom callback: logs episode metrics (reward, lap time, etc.) to CSV.
        MKDSMetricsCallback(log_dir=f"{base_path}/logs"),

        # Periodic checkpoint: saves model weights every save_freq steps.
        # save_replay_buffer=True is critical for off-policy DQN -- without it,
        # resuming training restarts with an empty buffer, causing a cold-start
        # quality drop that can last tens of thousands of steps.
        CheckpointCallback(save_freq=args.save_freq, save_path=f"{base_path}/models/",
                           name_prefix="mkds_ckpt", save_replay_buffer=True)
    ])

    try:
        logger.info("Training started. Press Ctrl+C to stop safely.")
        model.learn(
            total_timesteps=config.TOTAL_TIMESTEPS,
            callback=callbacks,
            # reset_num_timesteps=False preserves the global step counter when
            # resuming, so TensorBoard plots remain continuous and checkpoint
            # filenames keep incrementing rather than resetting to 0.
            reset_num_timesteps=False,
            tb_log_name=run_id,
        )
    except KeyboardInterrupt:
        logger.warning("Caught Ctrl+C. Saving current progress...")
    finally:
        # --- Safety save (always runs, even after KeyboardInterrupt) ---
        # Writes the current model and replay buffer before the process exits
        # so no training progress is lost regardless of when Ctrl+C was pressed.
        final_save = f"{base_path}/models/interrupted_exit"
        model.save(final_save)
        model.save_replay_buffer(f"{final_save}_replay_buffer")
        logger.info(f"Safety Save Complete: {final_save}")
        try:
            logger.info("Closing environments...")
            env.close()
        except (BrokenPipeError, EOFError, ConnectionResetError):
            # Subprocess workers may have already exited (e.g., if a worker
            # crashed during training), so pipe/socket errors are expected here
            # and can be safely ignored -- the OS will reclaim the processes.
            logger.warning("Environments already closed or pipe broken. Finalizing exit.")


if __name__ == "__main__":
    train()
