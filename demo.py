import os
import glob
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from env.mkds_gym_env import MKDSEnv
from src.utils import config

def select_model():
    """Scans for .zip files and lets the user choose one."""
    # Search in outputs root and the sb3_models folder recursively
    search_pattern = os.path.join("outputs", "**", "*.zip")
    model_files = glob.glob(search_pattern, recursive=True)

    if not model_files:
        print("No .zip models found in the /outputs directory.")
        return None

    print("\n--- Available Models ---")
    for i, file_path in enumerate(model_files, 1):
        # Print just the filename and its immediate parent folder for clarity
        display_name = os.path.relpath(file_path, "outputs")
        print(f"{i}) {display_name}")

    while True:
        try:
            choice = int(input(f"\nSelect a model to run (1-{len(model_files)}): "))
            if 1 <= choice <= len(model_files):
                selected_path = model_files[choice - 1]
                # Remove .zip extension because SB3 load appends it automatically
                return os.path.splitext(selected_path)[0]
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a valid number.")

def run_demo():
    model_path = select_model()
    if not model_path:
        return

    print(f"\nInitializing Mario Kart DS Environment with model: {os.path.basename(model_path)}...")
    
    # Force visualize=True for demo
    base_env = MKDSEnv(visualize=True)
    env = DummyVecEnv([lambda: base_env])
    env = VecFrameStack(env, n_stack=config.STACK_SIZE, channels_order='last')

    try:
        # Load the model
        # Using device="cpu" is often faster for single-env inference to avoid VRAM overhead
        model = DQN.load(model_path, env=env, device="cuda")
        print("DQN Model loaded successfully.")
    except Exception as e:
        print(f"Error: Model didn't load properly. {e}")
        base_env.emu.destroy()
        return

    obs = env.reset()
    episode_count = 1
    current_episode_reward = 0

    print(f"\n--- Starting Episode {episode_count} ---")
    print("Focus the SDL Window to see the agent drive.")
    print("Press Ctrl+C in this terminal to stop.")
    
    try:
        while True: 
            # Predict action - deterministic=True is standard for testing/demo
            action, _states = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, rewards, dones, infos = env.step(action)           
            current_episode_reward += rewards[0] 

            if dones[0]:
                print(f"Episode {episode_count} Finished | Reward: {current_episode_reward:.2f}")
                episode_count += 1
                current_episode_reward = 0
                print(f"\n--- Starting Episode {episode_count} ---")
                
    except KeyboardInterrupt:
        print("\nDemonstration stopped by user.")
    finally:
        base_env.emu.destroy() 
        print("Emulator closed.")

if __name__ == "__main__":
    run_demo()