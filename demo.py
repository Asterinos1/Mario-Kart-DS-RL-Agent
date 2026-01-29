import os
import time
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from env.mkds_gym_env import MKDSEnv
from src.utils import config

def run_demo():
    print("Initializing Mario Kart DS Environment (Continuous Loop)...")
    base_env = MKDSEnv()
    env = DummyVecEnv([lambda: base_env])
    env = VecFrameStack(env, n_stack=config.STACK_SIZE, channels_order='last')

    model_path = "outputs/mkds_dqn_final"
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: {model_path}.zip not found.")
        return
    try:
        model = DQN.load(model_path, env=env)
        print("DQN Model loaded.")
    except Exception:
        print("Model didn't load properly. Exiting..")

    obs = env.reset()
    episode_count = 1
    current_episode_reward = 0

    print(f"\n--- Starting Episode {episode_count} ---")
    print("Press Ctrl+C to stop the program.")
    
    try:
        while True: 
            # Predict action
            action, _states = model.predict(obs, deterministic=False)
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