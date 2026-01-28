import os
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from env.mkds_gym_env import MKDSEnv

def make_env():
    return MKDSEnv()

if __name__ == "__main__":
    # 1. Environment Setup
    num_envs = 1
    env = SubprocVecEnv([make_env for _ in range(num_envs)])
    env = VecFrameStack(env, n_stack=4, channels_order='last')

    # 2. Define Paths
    model_path = "outputs/mkds_dqn_final.zip"
    buffer_path = "outputs/mkds_dqn_replay_buffer.pkl"

    # 3. Model Setup for RTX 4060
    policy_kwargs = dict(
        net_arch=[512, 512],
        activation_fn=torch.nn.ReLU
    )

    # 4. Load or Initialize
    if os.path.exists(model_path):
        print(f"--- Found existing model: {model_path}. Resuming... ---")
        # Load the model and connect it to the current environment
        model = DQN.load(model_path, env=env, device="cuda")
        
        # Load the replay buffer if it exists to prevent "memory loss"
        if os.path.exists(buffer_path):
            print("--- Loading Replay Buffer ---")
            model.load_replay_buffer(buffer_path)
    else:
        print("--- No existing model found. Starting fresh training. ---")
        model = DQN(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            buffer_size=50000,
            learning_rate=0.0001,
            batch_size=128,
            target_update_interval=1000,
            exploration_fraction=0.1,
            verbose=1,
            device="cuda",
            tensorboard_log="./logs/"
        )

    # 5. Checkpoints with Replay Buffer Support
    # save_replay_buffer=True ensures your backups also have the memories
    checkpoint_callback = CheckpointCallback(
        save_freq=5000, 
        save_path='./outputs/sb3_models/',
        save_replay_buffer=True
    )

    print("Starting Parallel Training...")
    try:
        # reset_num_timesteps=False is critical to continue the learning curve/epsilon decay
        model.learn(
            total_timesteps=1000000, 
            callback=checkpoint_callback,
            reset_num_timesteps=False 
        )
    except KeyboardInterrupt:
        print("Interrupted by user. Saving current progress...")
    finally:
        # 6. Final Save (Model + Buffer)
        model.save(model_path)
        model.save_replay_buffer(buffer_path)
        print(f"Final model saved to {model_path}")