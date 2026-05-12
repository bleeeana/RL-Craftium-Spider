import gymnasium as gym
import craftium
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from ppo import PPOAgent
import torch
import random

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class Solver:
    def __init__(self, env_name:str = "Craftium/SpidersAttack-v0", episodes: int = 2000):
        self.scores = []
        
        self.env_name = env_name
        self.episodes = episodes
        self.agent = PPOAgent(action_size=10)
        self.episode_rewards = []
    
    def run(self, save_best : bool = False, writer: SummaryWriter = None):
        env = gym.make(self.env_name, render_mode="rgb_array")
        env = gym.wrappers.FrameStack(env, num_stack=4)
        best_score = 0
        for ep in range(self.episodes):
            state, _ = env.reset()
            #print(state.shape)
            episode_reward = 0
            while True:
                value, action, log_prob = self.agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                self.agent.train(state,reward, value, log_prob, done, action, writer)
                state = next_state
                if done:
                    if writer:
                        writer.add_scalar("Metrics/Episode Reward", episode_reward, ep)
                    self.episode_rewards.append(episode_reward)
                    avg = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >=50 else np.mean(self.episode_rewards)
                    if ep % 50 == 0:
                        print(f"Episode {ep:3d} | Score: {episode_reward:3.0f} | AVG score(50): {avg:0.2f}")
                    break
            if save_best and avg > best_score:
                best_score = avg
                self.agent.save("spider-attack.pth")
                
        env.close()
         
        
    def test(self, episodes=10, render: bool = True, record: bool = False, video_folder: str = None):
        pass
    
    def run_experiments(self, episodes=1000):
        pass
        
    
def main():
    exp_name = f"ppo__steps={2000}_ppoepochs={3}_updateperiod={2048}"
    writer = SummaryWriter(log_dir=f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}/{exp_name}")
    solver = Solver()
    solver.run(True, writer)
    
if __name__ == "__main__":
    main()