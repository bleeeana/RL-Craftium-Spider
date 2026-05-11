import numpy as np
import torch
from collections import deque
import random
from nn import Actor, Critic, CNNEncoder
import torch.nn.functional as F

class Memory:
    def __init__(self):
        self.clear()
    
    def append_new(self, state, action,reward, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []
        
    def compute_returns(self, gamma: float):
        returns = []
        reward = 0
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                reward = 0
            reward = r + gamma * reward
            returns.insert(0, reward)
        return returns
    
    def compute_gae(self, gamma, gae_lambda):
        pass
    
    def compute_advantage(self,returns, values):
        return np.array(returns) - np.array(values)
    
    def compute_normalized_advantage(self, returns, values):
        advantage = self.compute_advantage(returns, values)
        return (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    
class PPOAgent:
    def __init__(self, state_size: int, action_size: int, lr: float = 3e-4,
                 gamma: float = 0.99, clip_ratio: float = 0.2, ppo_epochs: int = 3,
                 hidden_size: int = 512, layers_num: int = 0, normalize: bool = False,
                 critic_loss_coeff: float = 0.3, update_period:int = 2048, frame_size: int = 4,
                 feature_size: int = 256, channels_size: int = 3, batch_size: int = 256, 
                 gae_lambda: int = 0.95):
        self.gamma = gamma
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.critic_loss_coeff = critic_loss_coeff
        self.update_period = update_period
        self.normalize = normalize
        self.actor = Actor(feature_size, hidden_size, action_size, layers_num).to(self.actor.devce)
        self.encoder = CNNEncoder(feature_size, channels_size, frame_size).to(self.actor.device)
        self.critic = Critic(feature_size, hidden_size, action_size, layers_num).to(self.actor.device)
        all_params = list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=lr)
        
        self.memory = Memory()
        
        self.steps = 0
        self.iteration = 0
        self.loss = 0
        
        self.entropy_loss_coeff = 0.05
    
    def ppo_loss(self, states_tensor, actions_tensor, advantages_tensor, returns_tensor, old_log_probs_tensor):
        
        logits = self.actor(states_tensor)
        distribution = torch.distributions.Categorical(logits)
        log_probs = distribution.log_prob(actions_tensor)
        entropy = distribution.entropy.mean()
        
        ratio = torch.exp(log_probs - old_log_probs_tensor)
        ratio_clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        values = self.critic(states_tensor)
        
        self.loss = -torch.min(ratio * advantages_tensor, ratio_clipped * advantages_tensor).mean() + \
            self.critic_loss_coeff * F.mse_loss(values, returns_tensor) - \
            self.entropy_loss_coeff * entropy
            
    def act(self, state):
        with torch.no_grad():
            features = self.encoder(state)
            log_probs, action =
        
    def train(self, state, reward, log_prob, done, action):
        self.steps += 1
        self.memory.append_new(state, action, reward, log_prob, done)
        
        if self.steps % self.update_period == 0:
            advantages, returns = self.memory.compute_gae(self.gamma, self.gae)
        
            states_array = np.array(self.memory.states)  
            states_tensor = torch.FloatTensor(states_array).to(self.actor.device)
            with torch.no_grad():
                _, values = self.actor_critic(states_tensor)
            values = values.cpu().numpy().squeeze()
            
            
            states_tensor = torch.FloatTensor(np.array(self.memory.states)).to(self.actor.device)
            actions_tensor = torch.FloatTensor(np.array(self.memory.actions)).unsqueeze(-1).to(self.actor.device)
            advantages_tensor = torch.FloatTensor(np.array(advantages)).to(self.actor.device)
            returns_tensor = torch.FloatTensor(np.array(returns)).to(self.actor.device)
            old_log_probs_tensor = torch.FloatTensor(np.array(self.memory.log_probs)).to(self.actor.device)
            for _ in range(self.ppo_epochs):
                
                self.ppo_loss(states_tensor, actions_tensor, advantages_tensor, returns_tensor, old_log_probs_tensor)
                self.optimizer.zero_grad()
                self.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)


                self.optimizer.step()
                
            self.memory.clear()

    
    def save(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    def load(self, path):
        model = torch.load(path, map_location=self.actor.device)
        self.encoder.load_state_dict(model['encoder_state_dict'])
        self.actor.load_state_dict(model['actor_state_dict'])
        self.critic.load_state_dict(model['critic_state_dict'])
        self.optimizer.load_state_dict(model['optimizer_state_dict'])