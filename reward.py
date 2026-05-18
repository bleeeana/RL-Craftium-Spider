import gymnasium as gym
import numpy as np

class ChangeRewardWrapper(gym.Wrapper):
    def __init__(self, env, attack_idx: int = 1, miss_penalty: float = -0.015,  
                 afk_penalty: float = -0.005, kill_bonus: float = 15.0,
                 mouse_x_plus_idx: int = 2, mouse_x_minus_idx: int = 3,
                 mouse_y_plus_idx: int = 4, mouse_y_minus_idx: int = 5,
                 jitter_penalty: float = -0.02, action_wrapper: str = 'binary',
                 still_steps_threshold: int = 15, extra_kill_bonus: int = 5,
                 time_bonus: float = 1.0, min_time_bonus: float = 0.3, 
                 time_bonus_decay = 0.999):
        super(ChangeRewardWrapper, self).__init__(env)
        self.attack_idx = attack_idx
        self.miss_penalty = miss_penalty
        self.afk_penalty = afk_penalty
        self.kill_bonus = kill_bonus
        self.mouse_x_plus_idx = mouse_x_plus_idx
        self.mouse_x_minus_idx = mouse_x_minus_idx
        self.mouse_y_plus_idx = mouse_y_plus_idx
        self.mouse_y_minus_idx = mouse_y_minus_idx
        self.jitter_penalty = jitter_penalty
        self.env = env
        self.attack_count = 0
        self.action_wrapper = action_wrapper
        self.still_steps_threshold = still_steps_threshold
        self.prev_info = None
        self.still_count = 0
        self.episode_kills = 0
        self.extra_kill_bonus = extra_kill_bonus
        self.time_bonus = time_bonus
        self.min_time_bonus = min_time_bonus
        self.time_bonus_decay = time_bonus_decay
        
    def step(self, action: np.ndarray):
        return self.step_binary(action) if self.action_wrapper == "binary" else self.step_discrete(action)

    def step_binary(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        bonus = 0
        if action.sum() == 0:
            bonus += self.afk_penalty
            
        if action[self.attack_idx] == 1:
            self.attack_count += 1
            if reward > 0:
                self.episode_kills += 1
                bonus += (self.kill_bonus + (self.episode_kills - 1) * self.extra_kill_bonus) * self.time_bonus
                self.time_bonus = 1.0
            else:
                self.miss_count += 1
                bonus += self.miss_penalty
            
        if action[self.mouse_x_plus_idx] == 1 and action[self.mouse_x_minus_idx] == 1:
            bonus += self.jitter_penalty
         
        if action[self.mouse_y_plus_idx] == 1 and action[self.mouse_y_minus_idx] == 1:
            bonus += self.jitter_penalty   
            
        if self.prev_info is not None:
            if np.allclose(info["player_pos"], self.prev_info["player_pos"]):
                self.still_count += 1
                if self.still_count > self.still_steps_threshold:
                    bonus += self.afk_penalty
            else:
                self.still_count = 0 
            
        new_reward = bonus + reward
        
        self.prev_info = info
        info["raw_reward"] = reward
        info['attack_count'] = self.attack_count
        info['miss_count'] = self.miss_count
        return state, new_reward, terminated, truncated, info
    
    def step_discrete(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        bonus = 0
        
        if action == 0:
            bonus += self.afk_penalty
    
        elif action[self.attack_idx] == 1:
            self.attack_count += 1
            if reward > 0:
                self.episode_kills += 1
                bonus += (self.kill_bonus + (self.episode_kills - 1) * self.extra_kill_bonus) * self.time_bonus
                self.time_bonus = 1.0
            else:
                self.miss_count += 1
                bonus += self.miss_penalty
            
        if self.prev_info is not None:
            if np.allclose(info["player_pos"], self.prev_info["player_pos"]):
                self.still_count += 1
                if self.still_count > self.still_steps_threshold:
                    bonus += self.afk_penalty
            else:
                self.still_count = 0 
                
        new_reward = bonus + reward
                
        self.time_bonus = max(self.min_time_bonus, self.time_bonus * self.time_bonus_decay)
        self.prev_info = info
        info["raw_reward"] = reward
        info["attack_count"] = self.attack_count
        info['miss_count'] = self.miss_count

        return state, new_reward, terminated, truncated, info
        
    def reset(self, **kwargs):
        self.attack_count = 0
        self.prev_info = None
        self.still_count = 0
        self.episode_kills = 0
        self.time_bonus = 1.0
        self.miss_count = 0
        return self.env.reset(**kwargs)