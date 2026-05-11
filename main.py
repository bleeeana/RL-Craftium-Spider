import gymnasium as gym
import craftium
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import torch
import random

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class Solver:
    def __init__(self, env_name: "Craftium/SpidersAttack-v0", episodes: int = 2000, video_episodes = 1000):
        self.scores = []
        
        self.env_name = env_name
        self.episodes = episodes
        self.video_episodes = video_episodes
    
    def run(self, save_best : bool = False, writer: SummaryWriter = None):
        pass
        
    def test(self, episodes=10, render: bool = True, record: bool = False, video_folder: str = None):
        pass
    
    def run_experiments(self, episodes=1000):
        pass
        
    
def main():
    pass
    
if __name__ == "__main__":
    main()