import os
import time
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from env.mkds_gym_env import MKDSEnv
from src.utils import config

def run_demo():
    print("Initializing Mario Kart DS Environment (Continuous Loop)...")
    
    # 1. Setup Environment with correct stacking
    base_env = MKDSEnv()
    env = DummyVecEnv([lambda: base_env])
    # n_stack=4 ensures shape (4, 84, 84) matches your trained model
    env = VecFrameStack(env, n_stack=config.STACK_SIZE, channels_order='last')

    model_path = "outputs/mkds_dqn_final"
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: {model_path}.zip not found.")
        return

    # 2. Load the Model
    try:
        model = PPO.load(model_path, env=env)
        print("PPO Model loaded.")
    except Exception:
        model = DQN.load(model_path, env=env)
        print("DQN Model loaded.")

    # 3. Execution Loop
    obs = env.reset()
    episode_count = 1
    current_episode_reward = 0

    print(f"\n--- Starting Episode {episode_count} ---")
    print("Press Ctrl+C to stop the program.")
    
    try:
        while True: # Infinite loop for continuous runs
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, rewards, dones, infos = env.step(action)
            
            current_episode_reward += rewards[0]
            
            # When an episode ends, SB3 automatically resets the env
            # The 'obs' returned above is already the FIRST frame of the NEW episode.
            if dones[0]:
                print(f"Episode {episode_count} Finished | Reward: {current_episode_reward:.2f}")
                
                # Reset local counters for the next run
                episode_count += 1
                current_episode_reward = 0
                print(f"\n--- Starting Episode {episode_count} ---")

    except KeyboardInterrupt:
        print("\nDemonstration stopped by user.")
    finally:
        base_env.emu.destroy() # Ensure DeSmuME process is killed
        print("Emulator closed.")

if __name__ == "__main__":
    run_demo()