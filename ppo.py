import numpy as np
import torch
from collections import deque
import random
from nn import Actor, Critic, CNNEncoder
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class Memory:
    def __init__(self, size_per_env: int, num_envs: int, action_wrapper: str, state_shape: tuple[int,int,int,int] = (4,64,64,3), action_size: int = 9):
        self.idx = 0
        self.size = size_per_env
        self.num_envs = num_envs
        self.state_shape = state_shape
        self.action_size = action_size
        self.action_wrapper = action_wrapper

        self.states = np.zeros((self.size, self.num_envs, *state_shape), dtype=np.float32) 
        self.actions = np.zeros((self.size, self.num_envs, action_size), dtype=np.float32) if action_wrapper == "binary" else np.zeros((self.size, self.num_envs), dtype=np.int64)
        self.rewards = np.zeros((self.size, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.size, self.num_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.size, self.num_envs), dtype=np.float32)
        self.values = np.zeros((self.size, self.num_envs), dtype=np.float32)
    
    def append_new(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, 
                   values: np.ndarray, log_probs: np.ndarray, dones: np.ndarray) -> None:
        self.states[self.idx] = states
        self.actions[self.idx] = actions
        self.rewards[self.idx] = rewards
        self.log_probs[self.idx] = log_probs
        self.dones[self.idx] = dones
        self.values[self.idx] = values
        self.idx += 1

    def clear(self) -> None:
        self.idx = 0
        
    def compute_gae(self, gamma: float, gae_lambda: float,
                    last_values: np.ndarray, last_dones: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        last_gae = np.zeros(self.num_envs, dtype=np.float32)
        advantages = np.zeros((self.size, self.num_envs), dtype=np.float32)
        
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_values = last_values
                next_non_terminal = 1.0 - last_dones
            else:
                next_values = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
                
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
            
        returns = advantages + self.values
        
        returns = returns.reshape(-1)
        advantages = advantages.reshape(-1)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
class PPOAgent:
    def __init__(self, action_size: int, lr: float = 2.5e-4,
                 gamma: float = 0.999, clip_ratio: float = 0.1, ppo_epochs: int = 4,
                 hidden_size: int = 512, layers_num: int = 0, action_wrapper: str = "binary",
                 critic_loss_coeff: float = 0.7, update_period: int = 2048, frame_size: int = 4,
                 feature_size: int = 256, channels_size: int = 3, batch_size: int = 256, 
                 gae_lambda: float = 0.95, entropy_loss_coeff: float = 0.01, entropy_coeff_decay: float = 0.995,
                 min_entropy: float = 0.001, use_scheduler: bool = False, updates: int = 4000,
                 target_kl : float = 0.015, num_envs: int = 4):
        self.gamma = gamma
        self.action_size = action_size
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.critic_loss_coeff = critic_loss_coeff
        self.update_period = update_period
        self.action_wrapper = action_wrapper

        self.actor = Actor(feature_size, hidden_size, action_size, layers_num)
        self.actor.to(self.actor.device)
        self.encoder = CNNEncoder(feature_size, channels_size, frame_size).to(self.actor.device)
        self.critic = Critic(feature_size, hidden_size, 1, layers_num).to(self.actor.device)
        all_params = list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=lr, eps=1e-5)
        
        self.memory = Memory(update_period // num_envs, num_envs, action_wrapper, action_size=action_size)
        
        self.steps = 0
        self.iteration = 0
        self.loss = 0
        self.entropy = 0
        self.start_lr = lr
        self.entropy_loss_coeff = entropy_loss_coeff
        self.entropy_coeff_decay = entropy_coeff_decay
        self.min_entropy = min_entropy
        self.use_scheduler = use_scheduler
        self.updates = updates
        self.target_kl = target_kl
        
    def ppo_loss(self, states_tensor: torch.Tensor, actions_tensor: torch.Tensor, 
                 advantages_tensor: torch.Tensor, returns_tensor: torch.Tensor, 
                 old_log_probs_tensor: torch.Tensor):
        features = self.encoder(states_tensor)
        logits = self.actor(features)
        if self.action_wrapper == "binary":
            distribution = torch.distributions.Bernoulli(logits=logits)
            log_probs = distribution.log_prob(actions_tensor).sum(dim=-1) 
            self.entropy = distribution.entropy().sum(dim=-1).mean()
        else: 
            distribution = torch.distributions.Categorical(logits=logits)
            log_probs = distribution.log_prob(actions_tensor.squeeze(-1).long())
            self.entropy = distribution.entropy().mean()
        ratio = torch.exp(log_probs - old_log_probs_tensor)
        ratio_clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        values = self.critic(features).squeeze(-1)
        
        actor_loss = -torch.min(ratio * advantages_tensor, ratio_clipped * advantages_tensor).mean() 
        critic_loss = self.critic_loss_coeff * F.smooth_l1_loss(values, returns_tensor)
        
        self.loss = actor_loss + critic_loss - self.entropy_loss_coeff * self.entropy

        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_probs + old_log_probs_tensor).mean()
            
        return distribution.probs, actor_loss, critic_loss, approx_kl
    
    def act(self, state: np.ndarray | torch.Tensor):
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.as_tensor(np.asarray(state), dtype=torch.float32).to(self.actor.device)
            
            features = self.encoder(state)
            log_probs, action = self.actor.action_probs(features, action_wrapper=self.action_wrapper)
            value = self.critic(features).squeeze(-1)
            if self.action_wrapper == "binary":
                action = action.cpu().numpy()
            else:
                action = action.cpu().numpy()
            return value.cpu().numpy(), action, log_probs.cpu().numpy()
            
    def estimate_value(self, state: np.ndarray | torch.Tensor) -> float:
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.as_tensor(np.asarray(state), dtype=torch.float32).to(self.actor.device)
            if state.dim() == 4:
                state = state.unsqueeze(0)
            features = self.encoder(state)
            return self.critic(features).cpu().numpy()
        
    def train(self, done: bool, writer: SummaryWriter = None, ep: int = 0,
              next_state: np.ndarray | torch.Tensor | None = None) -> None:

        last_value = self.estimate_value(next_state)
        advantages, returns = self.memory.compute_gae(self.gamma, self.gae_lambda, last_value, done)
        states_tensor = torch.from_numpy(self.memory.states.reshape(-1, *self.memory.state_shape)).float().to(self.actor.device)
        if self.action_wrapper == "binary":
            actions_tensor = torch.from_numpy(self.memory.actions.reshape(-1, self.action_size)).to(self.actor.device)
        else:
            actions_tensor = torch.from_numpy(self.memory.actions.reshape(-1)).to(self.actor.device) 
        advantages_tensor = torch.from_numpy(advantages).float().to(self.actor.device)
        returns_tensor = torch.from_numpy(returns).float().to(self.actor.device)
        old_log_probs_tensor = torch.from_numpy(self.memory.log_probs.reshape(-1)).float().to(self.actor.device)
        for _ in range(self.ppo_epochs):
            idx = np.random.permutation(len(states_tensor))
            early_stop = False
            for start in range(0,len(states_tensor), self.batch_size):
                batch_idx = idx[start: start + self.batch_size]
                
                probs, actor_loss, critic_loss, approx_kl = self.ppo_loss(states_tensor[batch_idx], actions_tensor[batch_idx], advantages_tensor[batch_idx], returns_tensor[batch_idx], old_log_probs_tensor[batch_idx])
                self.optimizer.zero_grad()
                self.loss.backward()
                all_params = list(self.actor.parameters()) + list(self.critic.parameters()) + list(self.encoder.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, 0.5)
                
                self.optimizer.step()

                if getattr(self, "target_kl", None) is not None and approx_kl.item() > self.target_kl:
                    early_stop = True
                    break

            if early_stop:
                break

        if writer:
            writer.add_scalar("Log/Advantage_mean", advantages.mean(), ep)
            writer.add_scalar("Log/Advantages_std", advantages.std(),ep)
            writer.add_scalar("Log/Entropy", self.entropy, ep)
            writer.add_scalar("Log/loss", self.loss, ep)
            writer.add_scalar("Log/actor_loss", actor_loss, ep)
            writer.add_scalar("Log/critic_loss", critic_loss, ep)

            writer.add_scalar("Log/Entropy_coeff", self.entropy_loss_coeff, ep)
            writer.add_histogram("Policy/probs", probs.detach().cpu(),ep)
            self.log_gradients(writer, ep)

        if self.use_scheduler:
            frac = 1.0 - (ep / self.updates) 
            current_lr = self.start_lr * frac

            self.change_learning_rate(current_lr)
            writer.add_scalar("Log/LR", current_lr, ep)

            
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


    def change_learning_rate(self, lr: float):
        for param in self.optimizer.param_groups:
            param['lr'] = lr
    
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