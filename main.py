import gymnasium as gym
import craftium
from craftium.wrappers import BinaryActionWrapper, DiscreteActionWrapper
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

SPIDER_ACTIONS = [
    "forward",
    "left",
    "right",
    "jump",
    "dig",
    "mouse x+",
    "mouse x-",
    "mouse y+",
    "mouse y-",
]

class Solver:
    def __init__(self, env_name:str = "Craftium/SpidersAttack-v0", episodes: int = 4000, env_type: str = "discrete"):
        self.scores = []
        self.env_type = env_type
        self.env_name = env_name
        self.episodes = episodes
        if self.env_type == "binary":
            action_size = len(SPIDER_ACTIONS)
        elif self.env_type == "discrete":
            action_size = len(SPIDER_ACTIONS) + 1
        else:
            raise ValueError(f"Unknown env_type: {self.env_type}")
        self.agent = PPOAgent(action_size=action_size, layers_num=2, action_wrapper=env_type)
        self.episode_rewards = []

    def run(self, save_best: bool = False, writer: SummaryWriter = None) -> None:
        env = self.make_env(render_mode=None)
        self.train_cycle(save_best, writer, env, self.episodes)


    def make_env(self, render_mode: str | None = None):
        return self.make_binary_env(render_mode) if self.env_type == "binary" else self.make_discrete_env(render_mode)

    def make_base_env(self, render_mode: str | None = None):
        kwargs = {
            "frameskip": 4,
            "fps_max": 200,
            "enable_voxel_obs": True
        }
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        return gym.make(self.env_name, **kwargs).unwrapped
    
    def make_binary_env(self, render_mode: str | None = None):
        env = self.make_base_env(render_mode)
        env = BinaryActionWrapper(
            env,
            actions=SPIDER_ACTIONS,
            mouse_mov=0.4
        )
        env = ChangeRewardWrapper(env, attack_idx=SPIDER_ACTIONS.index("dig"), action_wrapper="binary")
        env = gym.wrappers.FrameStack(env, num_stack=4)
        return env
    
    def make_discrete_env(self, render_mode: str | None = None):
        env = self.make_base_env(render_mode)
        env = DiscreteActionWrapper(
            env,
            actions=SPIDER_ACTIONS,
            mouse_mov=0.4
        )
        
        env = ChangeRewardWrapper(env, attack_idx=SPIDER_ACTIONS.index("dig") + 1, action_wrapper="discrete")
        env = gym.wrappers.FrameStack(env, num_stack=4)
        return env

    def fine_tune(self, path: str = "spider-attack.pth", save_best: bool = False,
                  writer: SummaryWriter = None, episodes: int = 2000, lr: float = 2e-4) -> None:
        self.agent.load(self.env_type + path)
        self.agent.change_learning_rate(lr)
        if self.env_type == "discrete":
            env = self.make_discrete_env()
        else:
            env = self.make_binary_env()
        self.train_cycle(save_best, writer, env, episodes, "finetuned-spider-attack.pth")

    def train_cycle(self, save_best: bool, writer: SummaryWriter, env, episodes: int, save_name: str = "spider-attack.pth"):
        best_score = -float('inf')
        checkpoint_path = self.env_type + save_name
        for ep in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_raw_reward = 0
            while True:
                value, action, log_prob = self.agent.act(state)
                if isinstance(action, np.ndarray):
                    action = action.astype(int) 
                next_state, new_reward, terminated, truncated, info = env.step(action)
                raw_reward = info["raw_reward"]
                attack_count = info["attack_count"]
                miss_count = info["miss_count"]
                episode_raw_reward += raw_reward
                episode_reward += new_reward
                done = terminated or truncated
                self.agent.train(state, new_reward, value, log_prob, done, action, writer, ep, next_state)
                state = next_state
                if done:
                    if writer:
                        writer.add_scalar("Metrics/Episode Reward", episode_reward, ep)
                        writer.add_scalar("Metrics/Episode Raw Reward", episode_raw_reward, ep)
                        writer.add_scalar("Log/Attack Count", attack_count, ep)
                        writer.add_scalar("Log/Miss Count", miss_count, ep)
                       
                    self.episode_rewards.append(episode_reward)
                    avg = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >=50 else np.mean(self.episode_rewards)
                    if ep % 50 == 0:
                        print(f"Episode {ep:3d} | Score: {episode_reward:3.0f} | AVG score(50): {avg:0.2f}")
                    break
            if save_best and avg > best_score:
                best_score = avg
                self.agent.save(checkpoint_path)
                
        env.close()
         
    def test(self, episodes: int = 10, render: bool = True, record: bool = False, 
             video_folder: str = None, checkpoint: str | None = None) -> None:
        if checkpoint is None:
            finetuned = Path(self.env_type + "finetuned-spider-attack.pth")
            checkpoint = str(finetuned if finetuned.exists() else Path(self.env_type + "spider-attack.pth"))
        self.agent.load(checkpoint)
        
        render_mode = "rgb_array" if record else ("human" if render else "rgb_array")
        env = self.make_env(render_mode=render_mode)
        
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
                if isinstance(action, np.ndarray):
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
    exp_name = f"ppo__steps={4000}_ppoepochs={4}_updateperiod={4096}_full_actions"
    writer = SummaryWriter(log_dir=f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}/{exp_name}")
    solver = Solver(env_type="discrete")
    solver.run(True, writer)
    solver.fine_tune(save_best=True, writer=writer)
    solver.test(10,record=True,render=False,video_folder='videos')
    
if __name__ == "__main__":
    main()