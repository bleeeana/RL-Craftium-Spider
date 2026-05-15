import numpy as np
import torch
from collections import deque
import random
from nn import Actor, Critic, CNNEncoder
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class Memory:
    def __init__(self,size: int, state_shape: tuple[int,int,int,int] = (4,64,64,3),action_size: int = 9):
        self.clear()
        self.idx = 0
        self.size = size
        self.states = np.zeros((size, *state_shape), dtype=np.uint8)
        self.actions = np.zeros((size, action_size), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
    
    def append_new(self, state: np.ndarray | torch.Tensor, action: np.ndarray, reward: float, 
                   value: float, log_prob: float, done: bool) -> None:
        self.states[self.idx] =state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.log_probs[self.idx] = log_prob
        self.dones[self.idx] = done
        self.values[self.idx] = value
        self.idx += 1

    def clear(self) -> None:
        self.idx = 0
        
    def compute_returns(self, gamma: float) -> list[float]:
        returns = []
        reward = 0
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                reward = 0
            reward = r + gamma * reward
            returns.insert(0, reward)
        return returns
    
    def compute_gae(self, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
        last_gae = 0.0
        advantages = np.zeros(self.size, dtype=np.float32)
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1 or self.dones[t]:
                next_value = 0.0
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0
                
            # TD-ошибка: r_t + gamma*V(s_t+1)*(1-done) - V(s_t)
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            
            # GAE: A_t = TD + gamma*gae_lambda*(1-done)*A_{t+1}
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
            
        returns = advantages + self.values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
        
    
    def compute_advantage(self, returns: np.ndarray | list[float], values: np.ndarray | list[float]) -> np.ndarray:
        return np.array(returns) - np.array(values)
    
    def compute_normalized_advantage(self, returns: np.ndarray, values: np.ndarray) -> np.ndarray:
        advantage = self.compute_advantage(returns, values)
        return (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    
class PPOAgent:
    def __init__(self, action_size: int, lr: float = 2.5e-4,
                 gamma: float = 0.99, clip_ratio: float = 0.2, ppo_epochs: int = 4,
                 hidden_size: int = 512, layers_num: int = 0, normalize: bool = False,
                 critic_loss_coeff: float = 0.5, update_period: int = 8192, frame_size: int = 4,
                 feature_size: int = 256, channels_size: int = 3, batch_size: int = 256, 
                 gae_lambda: float = 0.95, entropy_loss_coeff: float = 0.00, entropy_coeff_decay: float = 0.995,
                 min_entropy: float = 0.000):
        self.gamma = gamma
        self.action_size = action_size
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.critic_loss_coeff = critic_loss_coeff
        self.update_period = update_period
        self.normalize = normalize
        self.actor = Actor(feature_size, hidden_size, action_size, layers_num)
        self.actor.to(self.actor.device)
        self.encoder = CNNEncoder(feature_size, channels_size, frame_size).to(self.actor.device)
        self.critic = Critic(feature_size, hidden_size, 1, layers_num).to(self.actor.device)
        all_params = list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=lr)
        
        self.memory = Memory(update_period, action_size=self.action_size)
        
        self.steps = 0
        self.iteration = 0
        self.loss = 0
        self.entropy = 0
        
        self.entropy_loss_coeff = entropy_loss_coeff
        self.entropy_coeff_decay = entropy_coeff_decay
        self.min_entropy = min_entropy
    
    def ppo_loss(self, states_tensor: torch.Tensor, actions_tensor: torch.Tensor, 
                 advantages_tensor: torch.Tensor, returns_tensor: torch.Tensor, 
                 old_log_probs_tensor: torch.Tensor) -> None:
        features = self.encoder(states_tensor)
        logits = self.actor(features)
        probs = torch.sigmoid(logits)
        distribution = torch.distributions.Bernoulli(probs)
        log_probs = distribution.log_prob(actions_tensor).sum(dim=-1, keepdim=True)
        self.entropy = distribution.entropy().mean()
        
        ratio = torch.exp(log_probs - old_log_probs_tensor)
        ratio_clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        values = self.critic(features).squeeze(-1)
        
        self.loss = -torch.min(ratio * advantages_tensor, ratio_clipped * advantages_tensor).mean() + \
            self.critic_loss_coeff * F.mse_loss(values, returns_tensor) #- self.entropy_loss_coeff * self.entropy
            
    def act(self, state: np.ndarray | torch.Tensor) -> tuple[float, int, float]:
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.as_tensor(np.asarray(state), dtype=torch.float32).to(self.actor.device)
            if state.dim() == 4:
                state = state.unsqueeze(0)
            
            features = self.encoder(state)
            log_probs, action = self.actor.action_probs(features)
            value = self.critic(features).item()
            return value, action.cpu().numpy().flatten(), log_probs.item()
            
        
    def train(self, state: np.ndarray | torch.Tensor, reward: float, value: float, 
              log_prob: float, done: bool, action: np.ndarray, writer: SummaryWriter = None, ep: int = 0) -> None:
        self.steps += 1
        self.memory.append_new(state, action, reward, value, log_prob, done)
        
        if self.steps % self.update_period == 0:
            
            advantages, returns = self.memory.compute_gae(self.gamma, self.gae_lambda)
            
                
            states_tensor = torch.from_numpy(self.memory.states).float().to(self.actor.device)
            actions_tensor = torch.from_numpy(self.memory.actions).to(self.actor.device)
            advantages_tensor = torch.from_numpy(advantages).float().to(self.actor.device)
            returns_tensor = torch.from_numpy(returns).float().to(self.actor.device)
            old_log_probs_tensor = torch.from_numpy(self.memory.log_probs).float().to(self.actor.device)
            for _ in range(self.ppo_epochs):
                idx = np.random.permutation(len(states_tensor))
                for start in range(0,len(states_tensor), self.batch_size):
                    batch_idx = idx[start: start + self.batch_size]
                    
                    self.ppo_loss(states_tensor[batch_idx], actions_tensor[batch_idx], advantages_tensor[batch_idx], returns_tensor[batch_idx], old_log_probs_tensor[batch_idx])
                    self.optimizer.zero_grad()
                    self.loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                    
                    self.optimizer.step()
            if writer:
                writer.add_scalar("Log/Advantage_mean", advantages.mean(), ep)
                writer.add_scalar("Log/Advantages_std", advantages.std(),ep)
                writer.add_scalar("Log/Entropy", self.entropy, ep)
                writer.add_scalar("Log/loss", self.loss, ep)
                writer.add_scalar("Log/Entropy_coeff", self.entropy_loss_coeff, ep)
                self.log_gradients(writer, ep)
                
            self.entropy_loss_coeff = max(self.min_entropy, 
                                          self.entropy_loss_coeff * self.entropy_coeff_decay)
                
            self.memory.clear()
            
    def log_gradients(self, writer: SummaryWriter, step: int) -> None:
        total_norm = 0.0
        for param in self.encoder.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        for param in self.actor.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        for param in self.critic.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        
        writer.add_scalar("Gradients/Total_Norm", total_norm ** 0.5, step)

    
    def save(self, path: str) -> None:
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    def load(self, path: str) -> None:
        model = torch.load(path, map_location=self.actor.device)
        self.encoder.load_state_dict(model['encoder_state_dict'])
        self.actor.load_state_dict(model['actor_state_dict'])
        self.critic.load_state_dict(model['critic_state_dict'])
        self.optimizer.load_state_dict(model['optimizer_state_dict'])