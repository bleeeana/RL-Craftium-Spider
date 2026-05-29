import gymnasium as gym
import numpy as np

class ChangeRewardWrapper(gym.Wrapper):
    def __init__(self, env, attack_idx: int = 1, miss_penalty: float = -0.05,  
                 afk_penalty: float = -0.01, kill_bonus: float = 4.5,
                 mouse_x_plus_idx: int = 5, mouse_x_minus_idx: int = 6,
                 mouse_y_plus_idx: int = 7, mouse_y_minus_idx: int = 8,
                 jitter_penalty: float = -0.02, action_wrapper: str = 'binary',
                 still_steps_threshold: int = 15, extra_kill_bonus: int = 1.5,
                 time_bonus: float = 1.5, min_time_bonus: float = 0.3, 
                 time_bonus_decay: float = 0.9995, death_penalty: float = -1.0,
                 attack_bonus: float = 0.05, attack_bonus_decay: float = 1.0,
                 min_attack_bonus: float = 0.003, miss_threshold: int = 8,
                 base_bonus: float = 0.000):
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
        self.death_penalty = death_penalty
        self.attack_bonus = attack_bonus
        self.attack_bonus_decay = attack_bonus_decay
        self.min_attack_bonus = min_attack_bonus
        self.miss_threshold = miss_threshold
        self.current_misses = 0
        self.base_bonus = base_bonus
        
    def step(self, action: np.ndarray):
        return self.step_binary(action) if self.action_wrapper == "binary" else self.step_discrete(action)

    def step_binary(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        bonus = 0
        if action.sum() == 0:
            bonus += self.afk_penalty
            
        if action[self.attack_idx] == 1:
            self.attack_count += 1
            bonus += self.attack_bonus
            if reward <= 0:
                self.miss_count += 1
                self.current_misses += 1
                if self.current_misses > self.miss_threshold:
                    bonus += self.miss_penalty

        if reward > 0:
            self.episode_kills += 1
            bonus += (self.kill_bonus + (self.episode_kills - 1) * self.extra_kill_bonus) * self.time_bonus
            self.time_bonus = 1.5
            self.current_misses = 0
            
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


        if terminated:
            bonus += self.death_penalty
            
        new_reward = bonus + reward
        
        self.time_bonus = max(self.min_time_bonus, self.time_bonus * self.time_bonus_decay)
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
    
        elif action == self.attack_idx:
            self.attack_count += 1
            bonus += self.attack_bonus
            if reward <= 0:
                self.miss_count += 1
                self.current_misses += 1
                if self.current_misses > self.miss_threshold:
                    bonus += self.miss_penalty

        elif action not in [0, self.attack_idx]:
            bonus += self.base_bonus

        if reward > 0:
            self.episode_kills += 1
            bonus += (self.kill_bonus + (self.episode_kills - 1) * self.extra_kill_bonus) * self.time_bonus
            self.time_bonus = 1.5
            self.current_misses = 0
            
        if self.prev_info is not None:
            if np.allclose(info["player_pos"], self.prev_info["player_pos"]):
                self.still_count += 1
                if self.still_count > self.still_steps_threshold:
                    bonus += self.afk_penalty
            else:
                self.still_count = 0 

        if terminated:
            bonus += self.death_penalty
                
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
        self.time_bonus = 1.5
        self.miss_count = 0
        self.current_misses = 0
        self.attack_bonus = max(self.min_attack_bonus, self.attack_bonus * self.attack_bonus_decay)
        return self.env.reset(**kwargs)
