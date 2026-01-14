import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_ # type: ignore
import gym
import argparse
import pygame
from teleop import collect_demos
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os


device = torch.device('cpu')


def collect_human_demos(num_demos: int):
    """
    Collect human demonstration data for the Mountain Car environment.
    
    Creates a Mountain Car environment and collects human-controlled demonstrations
    using keyboard input. The user controls the car with LEFT and RIGHT arrow keys,
    with no-op as the default action.
    
    Args:
        num_demos (int): The number of demonstration episodes to collect.
    
    Returns:
        list: A list of collected demonstrations, where each demonstration contains
              the sequence of states, actions, and rewards from a single episode.
    
    Note:
        - LEFT arrow key maps to action 0 (push left)
        - RIGHT arrow key maps to action 2 (push right)
        - Action 1 (no-op) is used as the default action when no key is pressed
    """
    mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 2}
    env = gym.make("MountainCar-v0",render_mode='single_rgb_array') 
    demos = collect_demos(env, keys_to_action=mapping,  # type: ignore
                          num_demos=num_demos, noop=1)
    return demos


def torchify_demos(sas_pairs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert demonstration state-action-state pairs to PyTorch tensors.

    This function takes a list of (state, action, next_state) tuples and converts
    them into PyTorch tensors on the specified device. States and next states are
    converted to float tensors, while actions are converted to long tensors.

    Args:
        sas_pairs (list): A list of tuples containing (state, action, next_state) pairs,
                          where states are numpy arrays and actions are integers.

    Returns:
        tuple: A tuple containing three PyTorch tensors:
            - obs_torch (torch.Tensor): Current states as float tensor on device.
            - acs_torch (torch.Tensor): Actions as long tensor on device.
            - obs2_torch (torch.Tensor): Next states as float tensor on device.
    """
    states = []
    actions = []
    next_states = []
    for s,a, s2 in sas_pairs:
        states.append(s)
        actions.append(a)
        next_states.append(s2)

    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)

    obs_torch: torch.Tensor = torch.from_numpy(np.array(states)).float().to(device)
    obs2_torch: torch.Tensor = torch.from_numpy(np.array(next_states)).float().to(device)
    acs_torch: torch.Tensor = torch.from_numpy(np.array(actions)).long().to(device)

    return obs_torch, acs_torch, obs2_torch


class PolicyNetwork(nn.Module):
    '''
        Simple neural network with two layers that maps a 2-d state to a prediction
        over which of the three discrete actions should be taken.
        The three outputs corresponding to the logits for a 3-way classification problem.

    '''
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        # so 2-dim input, output to classify into 3 actions
        """create the layers for the neural network. A two-layer network should be sufficient"""
        self.fc1 = nn.Linear(2, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 3)


    def forward(self, x):
        """this method performs a forward pass through the network, applying a non-linearity (ReLU is fine) on the hidden layers and should output logit values (since this is a discrete action task) for the 3-way classification problem"""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
   

def train_policy(obs: torch.Tensor, acs: torch.Tensor, 
                 nn_policy: PolicyNetwork, num_train_iters: int):
    # RH: lr is the learning rate
    print("Training policy with behavior cloning...")
    optimizer = Adam(nn_policy.parameters(), lr=1e-1)
    for _ in range(num_train_iters):
        logits = nn_policy(obs)
        loss = F.cross_entropy(logits, acs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return


def save_demos(demos, filename='demos.pkl'):
    """
    Save collected demonstrations to disk.
    
    Args:
        demos: The collected demonstrations to save.
        filename (str): Name of the file to save demos to. Defaults to 'demos.pkl'.
    """
    with open(filename, 'wb') as f:
        pickle.dump(demos, f)
    print(f"Demos saved to {filename}")


def load_demos(filename='demos.pkl'):
    """
    Load demonstrations from disk.
    
    Args:
        filename (str): Name of the file to load demos from. Defaults to 'demos.pkl'.
    
    Returns:
        The loaded demonstrations.
    """
    with open(filename, 'rb') as f:
        demos = pickle.load(f)
    print(f"Demos loaded from {filename}")
    return demos


def save_model(model, filename='policy_model.pt'):
    """
    Save trained model to disk.
    
    Args:
        model: The PyTorch model to save.
        filename (str): Name of the file to save model to. Defaults to 'policy_model.pt'.
    """
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


def load_model(filename='policy_model.pt'):
    """
    Load trained model from disk.
    
    Args:
        filename (str): Name of the file to load model from. Defaults to 'policy_model.pt'.
    
    Returns:
        PolicyNetwork: The loaded model.
    """
    model = PolicyNetwork()
    model.load_state_dict(torch.load(filename))
    model.eval()
    print(f"Model loaded from {filename}")
    return model
    
 
#evaluate learned policy
def evaluate_policy(pi, num_evals, human_render=True):
    if human_render:
        env = gym.make("MountainCar-v0",render_mode='human') 
    else:
        env = gym.make("MountainCar-v0") 

    policy_returns = []
    for i in range(num_evals):
        done = False
        total_reward = 0
        obs = env.reset()
        while not done:
            #take the action that the network assigns the highest logit value to
            #Note that first we convert from numpy to tensor and then we get the value of the 
            #argmax using .item() and feed that into the environment
            action = torch.argmax(pi(torch.from_numpy(obs).unsqueeze(0))).item()
            # print(action)
            obs, rew, done, info = env.step(action) # type: ignore
            total_reward += rew
        print("reward for evaluation", i, total_reward)
        policy_returns.append(total_reward)

    print("average policy return", np.mean(policy_returns))
    print("min policy return", np.min(policy_returns))
    print("max policy return", np.max(policy_returns))


def rh_main(num_demos: int, num_bc_iters: int, num_evals: int):
    demo_file = 'mountain_car_demos.pkl'
    if os.path.exists(demo_file):
        print(f"Found existing demos file: {demo_file}")
        demos = load_demos(demo_file)
    else:
        print(f"No existing demos found. Collecting {num_demos} new demos...")
        demos = collect_human_demos(num_demos)
        save_demos(demos, demo_file)
    
    obs, acs, _ = torchify_demos(demos)
    pi = PolicyNetwork()
    train_policy(obs, acs, pi, num_bc_iters)
    save_model(pi, 'mountain_car_policy.pt')
    evaluate_policy(pi, num_evals)
    return

# if __name__ == "__main__":
def main_argparse():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_demos', default = 1, type=int, help="number of human demonstrations to collect")
    parser.add_argument('--num_bc_iters', default = 100, type=int, help="number of iterations to run BC")
    parser.add_argument('--num_evals', default=6, type=int, help="number of times to run policy after training for evaluation")

    args = parser.parse_args()

    demo_file = 'mountain_car_demos.pkl'
    if os.path.exists(demo_file):
        print(f"Found existing demos file: {demo_file}")
        demos = load_demos(demo_file)
    else:
        print(f"No existing demos found. Collecting {args.num_demos} new demos...")
        demos = collect_human_demos(args.num_demos)
        save_demos(demos, demo_file)
    
    obs, acs, _ = torchify_demos(demos)
    pi = PolicyNetwork()
    train_policy(obs, acs, pi, args.num_bc_iters)
    save_model(pi, 'mountain_car_policy.pt')
    evaluate_policy(pi, args.num_evals)
    return

if __name__ == "__main__":
    rh_main(num_demos=5, num_bc_iters=100, num_evals=6)
