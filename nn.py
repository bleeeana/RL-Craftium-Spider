import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d
import numpy as np

class CNNEncoder(nn.Module):
    def __init__(self, output_size: int, channels_size: int, frame_size: int):
        super(CNNEncoder, self).__init__() 
        self.layers = nn.ModuleList([
            # batch 12 64 64
            Conv2d(channels_size * frame_size, 32, 8, stride=4), nn.ReLU(),
            # batch 32 15 15
            Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            # batch 64 6 6
            Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            # batch 64 4 4
            nn.Flatten(),
            # batch 1024
            nn.Linear(1024,output_size),nn.ReLU()
        ])
        for m in self.modules():
            if isinstance(m, (Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, frames):
        
        frames = frames.permute(0,4,1,2,3).reshape(frames.shape[0], -1, 64, 64)
        frames /=255.0
        for layer in self.layers:
            frames = layer(frames)
            
        return frames
    
class Actor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, action_size: int, layers_num: int,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(Actor, self).__init__()
        actual_size = input_size
        self.layers = nn.ModuleList([])
        for i in range(layers_num):
            self.layers.append(nn.Linear(actual_size, hidden_size))
            actual_size = hidden_size
        self.device = device
        self.activation = nn.ReLU()
        self.layers.append(nn.Linear(actual_size, action_size))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
                
    def forward(self, state: torch.Tensor):
        for layer in self.layers[:-1]:
            state = self.activation(layer(state))
        state = self.layers[-1](state)
        return state
        
    def action_probs(self, state: torch.Tensor):
        with torch.no_grad():
            logits = self.forward(state)
            probs = F.softmax(logits, dim=-1)
            distribution = torch.distributions.Categorical(probs)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            return log_prob, action 
        
class Critic(nn.Module):
    def __init__(self, input_size: int = 256, hidden_size: int = 512, action_size: int = 1, layers_num: int = 0):
        super(Critic, self).__init__()
        actual_size = input_size
        self.layers = nn.ModuleList([])
        for i in range(layers_num):
            self.layers.append(nn.Linear(actual_size, hidden_size))
            actual_size = hidden_size
        self.activation = nn.ReLU()
        self.layers.append(nn.Linear(actual_size, action_size))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
                
    def forward(self, state: torch.Tensor):
        for layer in self.layers[:-1]:
            state = self.activation(layer(state))
        value = self.layers[-1](state).squeeze(-1)
        return value