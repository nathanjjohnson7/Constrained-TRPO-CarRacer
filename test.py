
import gymnasium as gym
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import numpy as np

import imageio

class Actor(nn.Module):
    def __init__(self, input_dims=10, hidden=128, n_actions=5):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_dims, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_actions)
        self.out.weight.data.mul_(0.1)
        self.out.bias.data.mul_(0.0)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.out(x)
        return x

#Welford's Online Algorithm: 
#adapted from: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
# Retrieve the mean, variance and sample variance from an aggregate
def finalize_aggregate_stats(aggregate):
    count, mean, M2 = aggregate[0], aggregate[1], aggregate[2]
    if count[0] == 1: #can't have an std with just one value
        return mean, None, None, None, None
    variance = M2 / count
    sample_variance = M2 / (count - 1)
    std = torch.sqrt(variance)
    sample_std = torch.sqrt(sample_variance)
    return mean, variance, sample_variance, std, sample_std


def test(seed, model_path="best_actor.pt", aggregate_path="best_aggregate_stats.pt"):
    model = Actor()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    aggregate = torch.load(aggregate_path)

    score = 0

    env = gym.make("CarRacing-v3", continuous=False)
    obs, info = env.reset(seed=seed)
    obs, reward, done, truncated, info = env.step(int(4))
    frame = info["data"][1:]

    current_images = []
    try:
        while not done and len(current_images)<2000:
            state = torch.tensor(frame, dtype=torch.float32)

            #Create a mask to skip indices 4 and 5 - the sine and cosine of the car angle relative to the trajectory of the track
            #We don't want to normalize these separately with a running mean and std since sine and cosine values should add to 1
            mask = torch.ones_like(state, dtype=torch.bool)
            mask[4] = False
            mask[5] = False

            #normalize other values using running mean and std of each value
            obs_mean, _, _, obs_std, _ = finalize_aggregate_stats(aggregate)
            if obs_std is not None:
                obs_std = obs_std.clamp(min=1e-8)
                state[mask] = (state[mask] - obs_mean[mask]) / obs_std[mask]

            else:
                state[mask] = state[mask] - obs_mean[mask]
            state = state.unsqueeze(0)

            logits = model(state)
            action = torch.argmax(logits, dim=-1)

            obs, reward, done, truncated, info = env.step(action.item())

            current_images.append(obs)

            frame = info["data"][1:]

            score+=reward
    finally:
        imageio.mimsave(f"car_{seed}_score_{int(score)}.mp4", current_images, fps=30)
        env.close()
        return score

if __name__ == "__main__":
    score = test(seed=None)
    print("Final Score: ", score)
