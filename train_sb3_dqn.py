import os
import argparse
from datetime import datetime
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from env.mkds_gym_env import MKDSEnv
from src.utils import config

def make_env(visualize=False):
    return lambda: MKDSEnv(visualize=visualize)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MKDS SB3 DQN Trainer")
    parser.add_argument("--load", type=str, help="Path to a .zip model to resume training")
    parser.add_argument("--run_name", type=str, default=None, help="Custom name for this training run")
    args = parser.parse_args()

    # Create a unique ID for logs
    run_id = args.run_name or f"DQN_{datetime.now().strftime('%m%d_%H%M')}"
    
    # Environment Setup (No window for speed)
    num_envs = 4
    env = SubprocVecEnv([make_env(visualize=False) for _ in range(num_envs)])
    env = VecFrameStack(env, n_stack=config.STACK_SIZE, channels_order='last')

    model_path = "outputs/mkds_dqn_final.zip"
    
    if args.load and os.path.exists(args.load):
        print(f"--- Resuming from: {args.load} ---")
        model = DQN.load(args.load, env=env, device="cuda")
    else:
        print(f"--- Starting Fresh Run: {run_id} ---")
        model = DQN(
            "CnnPolicy", env,
            buffer_size=config.MEMORY_SIZE,        
            batch_size=config.BATCH_SIZE,         
            learning_rate=config.LEARNING_RATE,    
            optimize_memory_usage=True,
            verbose=1,
            device="cuda",
            tensorboard_log="./logs/" #
        )

    # Automated Checkpoints with Replay Buffers
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path=f'./outputs/sb3_models/{run_id}/',
        name_prefix="mkds_model",
        save_replay_buffer=True
    )

    try:
        model.learn(
            total_timesteps=1000000, 
            callback=checkpoint_callback,
            tb_log_name=run_id, # Segregates metrics in TensorBoard
            reset_num_timesteps=False 
        )
    finally:
        model.save(f"outputs/{run_id}_final")