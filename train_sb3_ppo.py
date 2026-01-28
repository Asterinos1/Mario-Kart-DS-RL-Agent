import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from env.mkds_env import MKDSEnv

def make_env():
    return MKDSEnv()

if __name__ == "__main__":
    # 1. Parallelize: Use 8 environments (half of your logical cores)
    num_envs = 8
    env = SubprocVecEnv([make_env for _ in range(num_envs)])
    
    # 2. Frame Stacking (SB3 version of your wrappers.py)
    env = VecFrameStack(env, n_stack=4, channels_order='last')

    # 3. Model Setup for RTX 4060
    # Enable mixed precision (Cuda/Tensor cores)
    policy_kwargs = dict(
        net_arch=[512, 512],
        activation_fn=torch.nn.ReLU
    )

    model = DQN(
        "CnnPolicy",
        env,
        buffer_size=50000,
        learning_rate=0.0001,
        batch_size=128, # Larger batch size for GPU efficiency
        target_update_interval=1000,
        exploration_fraction=0.1,
        verbose=1,
        device="cuda", # Force use of RTX 4060
        tensorboard_log="./logs/"
    )

    # 4. Save frequently
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./outputs/sb3_models/')

    print("Starting Parallel Training...")
    model.learn(total_timesteps=1000000, callback=checkpoint_callback)
    model.save("outputs/mkds_dqn_final")