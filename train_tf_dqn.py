import os
import tensorflow as tf
from env.mkds_custom_env import MKDSEnvLegacy 
from src.utils.wrappers import FrameStacker
from src.agents.dqn_agent import DQNAgent
from src.utils import config

def train():
    env = MKDSEnvLegacy() # Use legacy class
    agent = DQNAgent()
    stacker = FrameStacker()
    epsilon = config.EPSILON_START
    total_steps = 0

    for episode in range(1, 10001):
        raw_frame = env.reset()
        state = stacker.reset(raw_frame)
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, epsilon)
            # Legacy env returns 3 values, not 5
            next_raw_frame, reward, done = env.step(action)
            next_state = stacker.append(next_raw_frame)
            
            agent.memory.append((state, action, reward, next_state, done))
            
            if total_steps > config.BATCH_SIZE and total_steps % 4 == 0:
                agent.train()
                
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            if epsilon > config.EPSILON_END:
                epsilon -= (config.EPSILON_START - config.EPSILON_END) / config.EPSILON_DECAY
        
        print(f"Episode: {episode} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.2f}")

        if episode % 10 == 0:
            agent.update_target_network()
            agent.model.save_weights("outputs/mkds_dqn_weights.h5")

if __name__ == "__main__":
    train()