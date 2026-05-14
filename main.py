import gymnasium as gym
import craftium
from craftium.wrappers import BinaryActionWrapper
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from ppo import PPOAgent
import torch
import random
from reward import ChangeRewardWrapper

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class Solver:
    def __init__(self, env_name:str = "Craftium/SpidersAttack-v0", episodes: int = 4000):
        self.scores = []
        
        self.env_name = env_name
        self.episodes = episodes
        self.agent = PPOAgent(action_size=9, layers_num=2)
        self.episode_rewards = []
    
    def run(self, save_best : bool = False, writer: SummaryWriter = None) -> None:
        env = gym.make(self.env_name, render_mode="rgb_array").unwrapped
        env = BinaryActionWrapper(
            env,
            actions=[
                "forward", "left", "right", "jump", "dig",                                
                "mouse x+", "mouse x-", "mouse y+", "mouse y-",               
            ],
            mouse_mov=0.4
        )
        env = ChangeRewardWrapper(env)
        env = gym.wrappers.FrameStack(env, num_stack=4)
        best_score = 0
        for ep in range(self.episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_raw_reward = 0
            while True:
                value, action, log_prob = self.agent.act(state)
                action = action.astype(int) 
                next_state, new_reward, terminated, truncated, info = env.step(action)
                raw_reward = info["raw_reward"]
                attack_count = info["attack_count"]
                episode_raw_reward += raw_reward
                episode_reward += new_reward
                done = terminated or truncated
                self.agent.train(state,new_reward, value, log_prob, done, action, writer, ep)
                state = next_state
                if done:
                    if writer:
                        writer.add_scalar("Metrics/Episode Reward", episode_reward, ep)
                        writer.add_scalar("Metrics/Episode Raw Reward", episode_raw_reward, ep)
                        writer.add_scalar("Log/Attack Count", attack_count, ep)
                    self.episode_rewards.append(episode_reward)
                    avg = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >=50 else np.mean(self.episode_rewards)
                    if ep % 50 == 0:
                        print(f"Episode {ep:3d} | Score: {episode_reward:3.0f} | AVG score(50): {avg:0.2f}")
                    break
            if save_best and avg > best_score:
                best_score = avg
                self.agent.save("spider-attack.pth")
                
        env.close()
         
    def test(self, episodes: int = 10, render: bool = True, record: bool = False, 
         video_folder: str = None) -> None:
        self.agent.load("spider-attack.pth")
        
        render_mode = "human" if render else "rgb_array"
        env = gym.make(self.env_name, render_mode=render_mode, fps_max=0, pmul=1.0).unwrapped
        env = BinaryActionWrapper(
            env,
            actions=[
                "forward", "left", "right", "jump", "dig",                                
                "mouse x+", "mouse x-", "mouse y+", "mouse y-",               
            ],
            mouse_mov=0.4
        )
        env = ChangeRewardWrapper(env)
        env = gym.wrappers.FrameStack(env, num_stack=4)
        
        if record:
            Path(video_folder).mkdir(parents=True, exist_ok=True)
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_folder,
                episode_trigger=lambda x: True,
                name_prefix="spider_test",
            )
            print(f"Видео будут сохранены в папку: {video_folder}")
            render = False  
        
        self.scores = []
        raw_scores = []  
        
        for ep in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            raw_reward_sum = 0
            step_count = 0            
            while True:
                _, action, _ = self.agent.act(state)
                action = action.astype(int) 
                next_state, reward, terminated, truncated, info = env.step(action)
                raw_reward = info.get("raw_reward", 0)
                raw_reward_sum += raw_reward
                done = terminated or truncated
                total_reward += reward
                step_count += 1
                state = next_state
                if render and not record:
                    env.render()
                if done:
                    break
            
            self.scores.append(total_reward)
            raw_scores.append(raw_reward_sum) 
        env.close()
    
def main():
    exp_name = f"ppo__steps={4000}_ppoepochs={4}_updateperiod={8192}"
    writer = SummaryWriter(log_dir=f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}/{exp_name}")
    solver = Solver()
    solver.run(True, writer)
    solver.test(10,record=True,render=False,video_folder='videos')
    
if __name__ == "__main__":
    main()