import gymnasium as gym
import numpy as np

class ChangeRewardWrapper(gym.Wrapper):
    def __init__(self, env, attack_idx: int = 4, attack_bonus: float = 0.05,  
                 afk_penalty: float = -0.02, kill_bonus: float = 15.0,
                 miss_penalty: float = -0.06, max_miss: int = 7, 
                 attack_bonus_decay: float = 0.997, min_attack_bonus = 0.002):
        super(ChangeRewardWrapper, self).__init__(env)
        self.attack_idx = attack_idx
        self.attack_bonus = attack_bonus
        self.afk_penalty = afk_penalty
        self.kill_bonus = kill_bonus
        self.env = env
        self.miss_count = 0
        self.miss_penalty = miss_penalty
        self.max_miss = max_miss
        self.attack_bonus_decay = attack_bonus_decay
        self.min_attack_bonus = min_attack_bonus
        self.attack_count = 0
        
    def step(self, action: np.ndarray):
        state, reward, terminated, truncated, info = self.env.step(action)
        bonus = 0
        if action.sum() == 0:
            bonus += self.afk_penalty
            
        if action[self.attack_idx] == 1:
            self.attack_count += 1
            if self.miss_count > self.max_miss and reward == 0:
                bonus += self.miss_penalty
                
            self.miss_count += not reward
            bonus += self.attack_bonus
            
        if reward > 0:
            bonus += self.kill_bonus
            self.miss_count = 0
            
        new_reward = bonus + reward
        
        info["raw_reward"] = reward
        info['attack_count'] = self.attack_count
        return state, new_reward, terminated, truncated, info
    
    
    def reset(self, **kwargs):
        self.miss_count = 0
        self.attack_bonus = max(self.min_attack_bonus, self.attack_bonus * self.attack_bonus_decay)
        self.attack_count = 0
        return self.env.reset(**kwargs)