import gymnasium as gym
import numpy as np

class ChangeRewardWrapper(gym.Wrapper):
    def __init__(self, env, attack_idx: int = 5, attack_bonus: float = 0.03,  
                 afk_penalty: float = -0.01, kill_bonus: int =5.0,
                 miss_penalty: float = -0.02, max_miss: int = 3):
        super(ChangeRewardWrapper, self).__init__(env)
        self.attack_idx = attack_idx
        self.attack_bonus = attack_bonus
        self.afk_penalty = afk_penalty
        self.kill_bonus = kill_bonus
        self.env = env
        self.miss_count = 0
        self.miss_penalty = miss_penalty
        self.max_miss = max_miss
        
    def step(self, action: int):
        state, reward, terminated, truncated, info = self.env.step(action)
        bonus = 0
        if action == 0:
            bonus += self.afk_penalty
            
        if action == self.attack_idx:
            if self.miss_count > self.max_miss and reward == 0:
                bonus += self.miss_penalty
                
            bonus += self.attack_bonus
            
        if reward > 0:
            bonus += self.kill_bonus
            self.miss_count = 0
        
            
        new_reward = bonus + reward
        
        info["raw_reward"] = reward
        return state, new_reward, terminated, truncated, info
    
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)