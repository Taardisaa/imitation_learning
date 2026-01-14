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
from mountain_car_bc import collect_human_demos, torchify_demos, \
    train_policy, PolicyNetwork, evaluate_policy, save_demos, load_demos
import os


device = torch.device('cpu')


def collect_random_interaction_data(num_iters: int):
    """Initialize inverse dynamics model with random data before BCO loop"""
    state_next_state = []
    actions = []
    env = gym.make('MountainCar-v0')
    for _ in range(num_iters):
        obs = env.reset()

        done = False
        while not done:
            a = env.action_space.sample()
            next_obs, reward, done, info = env.step(a)  # type: ignore
            state_next_state.append(np.concatenate((obs,next_obs), axis=0)) # type: ignore
            actions.append(a)
            obs = next_obs
    env.close()
    return np.array(state_next_state), np.array(actions)


def collect_policy_interaction_data(policy: PolicyNetwork, num_iters: int):
    """
    Collect state transitions using current policy π_φ
    Corresponds to line 6, 7 in BCO algorithm.
    For BCO(0), this is only done in the first iteration of the BCO loop.
    """
    state_next_state = []
    actions = []
    env = gym.make('MountainCar-v0')
    
    for _ in range(num_iters):
        obs = env.reset()
        done = False
        
        while not done:
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action_probs = policy(obs_tensor)
                a = torch.argmax(action_probs).item()
            
            next_obs, reward, done, info = env.step(a)  # type: ignore
            state_next_state.append(np.concatenate((obs, next_obs), axis=0))  # type: ignore
            actions.append(a)
            obs = next_obs
            
    env.close()
    return np.array(state_next_state), np.array(actions)


class InvDynamicsNetwork(nn.Module):
    '''
        Neural network with that maps (s,s') state to a prediction
        over which of the three discrete actions was taken.
        The network should have three outputs corresponding to the logits for a 3-way classification problem.

    '''
    def __init__(self):
        super().__init__()

        #This network should take in 4 inputs corresponding to car position and velocity in s and s'
        # and have 3 outputs corresponding to the three different actions
        self.fc1 = nn.Linear(4, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 3)


    def forward(self, x):
        #this method performs a forward pass through the network
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    

def train_inv_dyn(inv_dyn: InvDynamicsNetwork, s_s2_torch: torch.Tensor, 
                  a_torch: torch.Tensor, num_iters: int=100,
                  learning_rate: float=1e-3):
    optimizer = Adam(inv_dyn.parameters(), lr=learning_rate)
    for _ in range(num_iters):
        optimizer.zero_grad()
        logits = inv_dyn(s_s2_torch)
        loss = F.cross_entropy(logits, a_torch)
        loss.backward()
        optimizer.step()
    return


# if __name__ == "__main__":
def main_argparse():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_demos', default = 1, type=int, help="number of human demonstrations to collect")
    parser.add_argument('--num_bc_iters', default = 100, type=int, help="number of iterations to run BC")
    parser.add_argument('--num_evals', default=6, type=int, help="number of times to run policy after training for evaluation")
    args = parser.parse_args()
    return main(args.num_demos, args.num_bc_iters, args.num_evals)

def main(num_demos: int, num_bc_iters: int, num_evals: int, 
         alpha: float = 0.0, num_bco_iters: int = 6):
    """
    I reformatted the code to match the lines in the BCO algorithm pseudocode.
    """
    # Line 1: Initialize the model M_θ as random approximator
    inv_dyn = InvDynamicsNetwork()
    
    num_interactions = 200
    s_s2, acs = collect_random_interaction_data(num_interactions)
    s_s2_torch = torch.from_numpy(np.array(s_s2)).float().to(device)
    a_torch = torch.from_numpy(np.array(acs)).long().to(device)
    train_inv_dyn(inv_dyn, s_s2_torch, a_torch, num_iters=100, learning_rate=1e-2)
    
    pi = PolicyNetwork()
    
    # Collect human demos once (Line 10: demonstrated state trajectories D_demo)
    demo_file = 'mountain_car_demos.pkl'
    if os.path.exists(demo_file):
        print(f"Found existing demos file: {demo_file}")
        demos = load_demos(demo_file)
        obs_demo, _, obs2_demo = torchify_demos(demos)
    else:
        print(f"No existing demos found. Collecting {num_demos} new demos...")
        demos = collect_human_demos(num_demos)
        save_demos(demos, demo_file)
        obs_demo, _, obs2_demo = torchify_demos(demos)
    
    # Line 3: Set I = number of episodes to collect per iteration
    I = num_interactions
    
    # fallback `policy improvement` to a fixed number of iterations
    s_s2_torch_ = None
    a_torch_ = None
    for iteration in range(num_bco_iters):
        print(f"Iters: {iteration+1}/{num_bco_iters}, I={I}")
        
        # Lines 5-8: for time-step t=1 to I do
        # Line 6-7: Generate samples and append to datasets using π_φ
        s_s2_pi, acs_pi = collect_policy_interaction_data(pi, I)
        s_s2_torch = torch.from_numpy(s_s2_pi).float().to(device)
        a_torch = torch.from_numpy(acs_pi).long().to(device)
        s_s2_torch_ = s_s2_torch if s_s2_torch_ is None else torch.cat((s_s2_torch_, s_s2_torch), dim=0)
        a_torch_ = a_torch if a_torch_ is None else torch.cat((a_torch_, a_torch), dim=0)
        
        # Line 9: Improve M_θ by modelLearning
        train_inv_dyn(inv_dyn, s_s2_torch_, a_torch_, num_iters=2000, learning_rate=1e-2)
        
        # Line 10-11: Use M_θ with T_demo to approximate A_demo
        state_trans_demo = torch.cat((obs_demo, obs2_demo), dim=1)
        outputs = inv_dyn(state_trans_demo)
        _, acs_pred = torch.max(outputs, 1)
        
        # Line 12: Improve π_φ by behavioralCloning(S_demo, A_demo)
        train_policy(obs_demo, acs_pred, pi, num_bc_iters)
        
        # Line 13: Set I = α|I^pre| (for BCO(0), α=0, so I becomes 0 after first iteration)
        I = int(alpha * num_interactions)
        if I == 0:
            print(f"I = 0 (BCO(0)), stopping after this iteration")
            break
    
    # Save trained models
    torch.save(inv_dyn.state_dict(), 'mountain_car_inv_dyn.pt')
    torch.save(pi.state_dict(), 'mountain_car_policy_bco.pt')
    print("\nModels saved: mountain_car_inv_dyn.pt, mountain_car_policy_bco.pt")
    
    # Evaluate learned policy
    print("\n=== Final Evaluation ===")
    evaluate_policy(pi, num_evals)


if __name__ == "__main__":
    # main_argparse()
    main(num_demos=10, num_bc_iters=100, num_evals=6, alpha=0)