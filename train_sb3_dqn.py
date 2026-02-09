import os
import glob
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from env.mkds_gym_env import MKDSEnv
from src.utils.callbacks import MKDSMetricsCallback
from src.utils import config

def select_resume_option():
    """Scans outputs directory and provides an interactive menu."""
    if not os.path.exists("outputs"): return None, None
    runs = [d for d in os.listdir("outputs") if os.path.isdir(os.path.join("outputs", d))]
    options = []
    for run in runs:
        model_files = glob.glob(f"outputs/{run}/models/*.zip")
        if model_files:
            latest_model = max(model_files, key=os.path.getmtime)
            options.append((run, latest_model))
    
    if not options: return None, None

    print("\n--- Available Models to Resume ---")
    for i, (run_id, path) in enumerate(options):
        print(f"{i}: {run_id} ({os.path.basename(path)})")
    
    choice = input(f"\nSelect index (Enter for NEW): ")
    return options[int(choice)] if choice.isdigit() and int(choice) < len(options) else (None, None)

def train():
    run_id, model_path = select_resume_option()
    
    # Setup environment (4 envs for Ryzen 7 7435HS overhead)
    env = SubprocVecEnv([lambda: MKDSEnv(visualize=False) for _ in range(4)])
    env = VecFrameStack(env, n_stack=config.STACK_SIZE, channels_order='last')

    if model_path:
        print(f"--- Resuming: {run_id} ---")
        model = DQN.load(model_path, env=env, device="cuda")
        # SB3 DQN requires manual replay buffer loading
        buffer_path = model_path.replace(".zip", "_replay_buffer.pkl")
        if os.path.exists(buffer_path):
            model.load_replay_buffer(buffer_path)
    else:
        run_id = f"DQN_{datetime.now().strftime('%m%d_%H%M')}"
        print(f"--- Fresh Run: {run_id} ---")
        model = DQN("CnnPolicy", env, verbose=1, device="cuda",
                    buffer_size=config.MEMORY_SIZE, batch_size=config.BATCH_SIZE)

    base_path = f"outputs/{run_id}"
    os.makedirs(f"{base_path}/models", exist_ok=True)
    os.makedirs(f"{base_path}/logs", exist_ok=True)

    # Combined Callbacks
    callbacks = CallbackList([
        MKDSMetricsCallback(log_dir=f"{base_path}/logs"),
        CheckpointCallback(save_freq=10000, save_path=f"{base_path}/models/", 
                           name_prefix="mkds_ckpt", save_replay_buffer=True)
    ])

    try:
        print("Training started. Press Ctrl+C to stop safely.")
        model.learn(total_timesteps=1000000, callback=callbacks, reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Caught Ctrl+C. Saving current progress...")
    finally:
        # Final Emergency Save
        final_save = f"{base_path}/models/interrupted_exit"
        model.save(final_save)
        model.save_replay_buffer(f"{final_save}_replay_buffer")
        print(f"Safety Save Complete: {final_save}")
        try:
            print("Closing environments...")
            env.close()
        except (BrokenPipeError, EOFError, ConnectionResetError):
            # On Windows, Ctrl+C often kills the sub-processes 
            # before we can send the 'close' command over the pipe.
            print("Environments already closed or pipe broken. Finalizing exit.")
if __name__ == "__main__":
    train()