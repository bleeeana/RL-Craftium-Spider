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
from collections import deque

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

AGENT_ACTIONS = [
    "forward",
    "left",
    "right",
    #"jump",
    "dig",
    "mouse x+",
    "mouse x-",
    "mouse y+",
    "mouse y-",
]

class Solver:
    def __init__(self, env_name:str = "Craftium/SpidersAttack-v0", updates: int = 1500, 
                 env_type: str = "discrete", num_envs: int = 4):
        self.scores = []
        self.env_type = env_type
        self.env_name = env_name
        self.updates = updates
        self.num_envs =num_envs
        if self.env_type == "binary":
            action_size = len(AGENT_ACTIONS)
        elif self.env_type == "discrete":
            action_size = len(AGENT_ACTIONS) + 1
        else:
            raise ValueError(f"Unknown env_type: {self.env_type}")
        self.agent = PPOAgent(action_size=action_size, layers_num=2, action_wrapper=env_type, updates=updates, num_envs=num_envs)

    def run(self, save_best: bool = False, writer: SummaryWriter = None) -> None:
        env_fns = [lambda i=i: self.make_env(i) for i in range(self.num_envs)]
        env = gym.vector.AsyncVectorEnv(env_fns)
        self.train_cycle(save_best, writer, env)


    def make_env(self, env_idx: int, render_mode: str | None = None):
        return self.make_binary_env(env_idx, render_mode) if self.env_type == "binary" else self.make_discrete_env(env_idx, render_mode)

    def make_base_env(self, env_idx: int,render_mode: str | None = None):
        kwargs = {
            "frameskip": 4,
            "fps_max": 200,
            "mt_port": 49155 + env_idx,
            "rgb_observations": True
        }
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        return gym.make(self.env_name, **kwargs).unwrapped
    
    def make_binary_env(self,env_idx: int, render_mode: str | None = None):
        env = self.make_base_env(env_idx, render_mode)
        env = BinaryActionWrapper(
            env,
            actions=AGENT_ACTIONS,
            mouse_mov=0.4
        )
        env = ChangeRewardWrapper(env, attack_idx=AGENT_ACTIONS.index("dig"),
                                    mouse_x_plus_idx=AGENT_ACTIONS.index("mouse x+"),
                                    mouse_x_minus_idx=AGENT_ACTIONS.index("mouse x-"),
                                    mouse_y_plus_idx=AGENT_ACTIONS.index("mouse y+"),
                                    mouse_y_minus_idx=AGENT_ACTIONS.index("mouse y-"),
                                    action_wrapper="binary")
        env = gym.wrappers.FrameStack(env, num_stack=4)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    
    def make_discrete_env(self,env_idx: int, render_mode: str | None = None):
        env = self.make_base_env(env_idx, render_mode)
        env = DiscreteActionWrapper(
            env,
            actions=AGENT_ACTIONS,
            mouse_mov=0.4
        )
        
        env = ChangeRewardWrapper(env, attack_idx=AGENT_ACTIONS.index("dig") + 1, action_wrapper="discrete")
        env = gym.wrappers.FrameStack(env, num_stack=4)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    def fine_tune(self, path: str = "spider-attack.pth", save_best: bool = False,
                  writer: SummaryWriter = None, updates: int = 2000, lr: float = 2.5e-4) -> None:
        self.agent.load(self.env_type + path)
        self.agent.change_learning_rate(lr)
        env_fns = [lambda i=i: self.make_env(i) for i in range(self.num_envs)]
        env = gym.vector.SyncVectorEnv(env_fns)
        self.updates = updates
        self.train_cycle(save_best, writer, env, "finetuned-spider-attack.pth")

    def train_cycle(self, save_best: bool, writer: SummaryWriter, env, save_name: str = "spider-attack.pth"):
        best_score = -float('inf')
        print(self.env_type, save_name)
        checkpoint_path = self.env_type + save_name
        global_step = 0
        state, _ = env.reset()
        env_steps = self.agent.update_period // self.num_envs
        
        self.ep_rewards_hist = deque(maxlen=50)

        for update in range(self.updates):
            for step in range(env_steps):
                global_step += self.num_envs
                value, action, log_prob = self.agent.act(state)
                
                if isinstance(action, np.ndarray) and self.env_type == "discrete":
                    action = action.astype(int)
                    
                next_state, reward, terminated, truncated, info = env.step(action)
                done = np.logical_or(terminated, truncated)
                
                self.agent.memory.append_new(state, action, reward * 0.1, value, log_prob, done)
                state = next_state

                if "final_info" in info:
                    for i, final_inf in enumerate(info["final_info"]):
                        if final_inf and "episode" in final_inf:
                            ep_reward = final_inf["episode"]["r"].item()
                            ep_raw = final_inf.get("raw_reward", 0)
                            ep_attacks = final_inf.get("attack_count", 0)
                            ep_misses = final_inf.get("miss_count", 0)
                            
                            self.ep_rewards_hist.append(ep_reward)
                            
                            if writer:
                                writer.add_scalar(f"Env_{i}/Episode_Reward", ep_reward, global_step)
                                writer.add_scalar(f"Env_{i}/Raw_Reward", ep_raw, global_step)
                                writer.add_scalar(f"Env_{i}/Attacks", ep_attacks, global_step)
                                writer.add_scalar(f"Env_{i}/Misses", ep_misses, global_step)
                                
                                #writer.add_histogram(f"Env_{i}/Reward_Distribution", ep_reward, global_step)

            if len(self.ep_rewards_hist) > 0:
                avg_reward = np.mean(self.ep_rewards_hist)
                
                print(f"Update {update:3d} | Step {global_step:6d} | AVG Score (for save): {avg_reward:5.1f}")
                
                if save_best and avg_reward > best_score:
                    best_score = avg_reward
                    self.agent.save(checkpoint_path)
                    
            self.agent.train(writer=writer, ep=update, next_state=state, done=done)
            
        env.close()

    def make_test_env(self, render_mode: str| None = None):
        kwargs = {
            "frameskip": 4,
            "fps_max": 200,
            "enable_voxel_obs": True,
            "mt_port": 49999, 
        }
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
            
        env = gym.make(self.env_name, **kwargs).unwrapped

        if self.env_type == "binary":
            env = BinaryActionWrapper(env, actions=AGENT_ACTIONS, mouse_mov=0.4)
            env = ChangeRewardWrapper(env, attack_idx=AGENT_ACTIONS.index("dig"),
                                        mouse_x_plus_idx=AGENT_ACTIONS.index("mouse x+"),
                                        mouse_x_minus_idx=AGENT_ACTIONS.index("mouse x-"),
                                        mouse_y_plus_idx=AGENT_ACTIONS.index("mouse y+"),
                                        mouse_y_minus_idx=AGENT_ACTIONS.index("mouse y-"),
                                        action_wrapper="binary")
        else:
            env = DiscreteActionWrapper(env, actions=AGENT_ACTIONS, mouse_mov=0.4)
            env = ChangeRewardWrapper(env, attack_idx=AGENT_ACTIONS.index("dig") + 1, action_wrapper="discrete")
            
        env = gym.wrappers.FrameStack(env, num_stack=4)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
         
    def test(self, episodes: int = 10, render: bool = True, record: bool = False, 
             video_folder: str = None, checkpoint: str | None = None) -> None:
        if checkpoint is None:
            finetuned = Path(self.env_type + "finetuned-spider-attack.pth")
            checkpoint = str(finetuned if finetuned.exists() else Path(self.env_type + "spider-attack.pth"))
        self.agent.load(checkpoint)
        
        render_mode = "rgb_array" if record else ("human" if render else "rgb_array")
        env = self.make_test_env(render_mode=render_mode)
        
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
                state_batch = np.expand_dims(np.array(state), 0)
                _, action, _ = self.agent.act(state_batch)
                env_action = action[0].astype(int) if self.env_type == "binary" else int(action[0])
                next_state, reward, terminated, truncated, info = env.step(env_action)
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
    exp_name = f"ppo_"
    writer = SummaryWriter(log_dir=f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}/{exp_name}")
    solver = Solver(env_type="discrete")
    #solver.run(True, writer)
    #solver.fine_tune(save_best=True, writer=writer)
    solver.test(10,record=True,render=False,video_folder='videos')
    
if __name__ == "__main__":
    main()