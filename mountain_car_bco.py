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
import tqdm
from pathlib import Path

# Use cuda for faster training if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collect_random_interaction_data(num_iters: int) -> tuple[np.ndarray, np.ndarray]:
    """Collect random state transitions from the environment. Will be later used to 
    initialize inverse dynamics model with random data **before** BCO loop.

    As I understand it, this can help the inverse dynamics model have 
    initial knowledge about the environment dynamics, rather than starting
    as a completely random approximator.

    Args:
        num_iters (int): number of episodes to collect. Increase this to get more data.
    
    Returns:
        state transitions and actions taken.
    """
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


# Added by Taardis
def collect_policy_interaction_data(policy: PolicyNetwork, num_iters: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Run current trained policy network to collect a trajectory of future state transitions.

    Args:
        policy (PolicyNetwork): current policy network to use for action selection.
        num_iters (int): number of episodes to collect. Increase this to get a longer trajectory.

    Returns:
        state transitions and actions taken.
    """
    state_next_state = []
    actions = []
    env = gym.make('MountainCar-v0')
    
    policy.eval()
    for _ in range(num_iters):
        obs = env.reset()
        done = False
        
        while not done:
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                acts = policy(obs_tensor)
                a = torch.argmax(acts).item()
            
            next_obs, reward, done, info = env.step(a)  # type: ignore
            state_next_state.append(np.concatenate((obs, next_obs), axis=0))  # type: ignore
            actions.append(a)
            obs = next_obs
            
    env.close()
    return np.array(state_next_state), np.array(actions)


class InvDynamicsNetwork(nn.Module):
    '''
        Neural network that maps (s,s') state to a prediction
        over which of the three discrete actions was taken.
        The network should have three outputs corresponding to the logits for a 3-way classification problem.
    '''
    def __init__(self, hidden_size=512):
        """
        NOTE: I once used a very straightforward and simple architecture here, which was a 4-layer fully connected
        network with relu. However that didn't work very well: the loss is always around 1.1 and accuracy is around 33%.
        
        So instead I changed to this one. I use batch-norm and dropout for better generalization.
        After testing, the loss soon converges to around 0, which is much better.
        
        Though its too much fancy and complex compared with the original one, but to make it work, I have no choice.
        """
        
        super().__init__()

        self.fc1 = nn.Linear(4, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x
    

def train_inv_dyn(inv_dyn: InvDynamicsNetwork, 
                  s_s2_torch: torch.Tensor, 
                  a_torch: torch.Tensor, num_iters: int=100,
                  learning_rate: float=1e-3, batch_size: int=128):
    """
    The function is to train the Inverse Dynamics Network, denoted as M_theta in the BCO paper.
    
    Args:
        inv_dyn (InvDynamicsNetwork): the inverse dynamics model to train.
        s_s2_torch (torch.Tensor): tensor of state transitions (s, s').
        a_torch (torch.Tensor): tensor of actions taken.
        num_iters (int, optional): number of training iterations. Defaults to 100.
        learning_rate (float, optional): learning rate for the optimizer. Defaults to 1e-3.
        batch_size (int, optional): batch size for training. Defaults to 128.
        
    NOTE: batching is implemented to improve model training.
    """
    optim = Adam(inv_dyn.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    inv_dyn.train()
    dataset_size = s_s2_torch.shape[0]
    with tqdm.tqdm(total=num_iters, desc="Training InvDyn") as pbar:
        for _ in range(num_iters):
            pbar.update(1)
            
            # get random permutation of indices for batching
            indices = torch.randperm(dataset_size)
            total_loss = 0
            num_batches = 0
            
            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i:min(i + batch_size, dataset_size)]
                batch_s_s2 = s_s2_torch[batch_indices]
                batch_a = a_torch[batch_indices]
                
                optim.zero_grad()
                logits = inv_dyn(batch_s_s2)
                loss = F.cross_entropy(logits, batch_a)
                loss.backward()
                
                optim.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})
        
    inv_dyn.eval()
    return


def main_argparse():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_demos', default = 1, type=int, help="number of human demonstrations to collect")
    parser.add_argument('--num_bc_iters', default = 100, type=int, help="number of iterations to run BC")
    parser.add_argument('--num_evals', default=6, type=int, help="number of times to run policy after training for evaluation")
    args = parser.parse_args()
    return bco_main(args.num_demos, args.num_bc_iters, args.num_evals)


def bco_main(num_demos: int, num_bc_iters: int, num_evals: int, 
         alpha: float = 0.0, num_bco_iters: int = 10, num_interactions: int =300):
    """
    The main function to run the BCO algorithm on MountainCar-v0 environment.
    
    Args:
        num_demos (int): number of human demonstrations to collect.
        num_bc_iters (int): number of behavioral cloning training iterations per BCO iteration.
        num_evals (int): number of evaluation episodes to run after training.
        alpha (float, optional): scaling factor for number of interactions in BCO. Defaults to 0.0 (BCO(0)).
        num_bco_iters (int, optional): number of BCO iterations to run. Defaults to 10. 
            In BCO(0), this will be treated as 1 since there will be no more post-demonstrations.
        num_interactions (int, optional): number of interactions to collect per BCO iteration. Defaults to 300.
    
    NOTE: I reformatted the code a bit to match the original BCO algorithm in the paper. 
    apologies for any confusion.
    """
    print(f"Using device: {device}")
    
    # Line 1: Initialize the model M_θ as random approximator
    inv_dyn = InvDynamicsNetwork(hidden_size=512).to(device)
    
    # hyperparameters for training InvDyn. This seems to be good enough after testing.  
    TRAIN_INV_ITERS = 50   
    TRAIN_INV_RATE = 5e-4
    
    # Train M_θ with initial random data but related to the game environment
    # NOTE: Originally I thought this isn't necessary, but after testing, it seems to help
    # the inverse dynamics model to have initial knowledge about the game environment.
    print(f"Collecting {num_interactions} random interaction data for initial InvDyn training...")
    s_s2, acs = collect_random_interaction_data(num_interactions)
    s_s2_torch = torch.from_numpy(np.array(s_s2)).float().to(device)
    a_torch = torch.from_numpy(np.array(acs)).long().to(device)
    print("Training initial inverse dynamics model...")
    train_inv_dyn(inv_dyn, s_s2_torch, a_torch, num_iters=TRAIN_INV_ITERS, learning_rate=TRAIN_INV_RATE)
    
    # Line 2: Initialize policy network. This is initialized as a fully random one.
    pi = PolicyNetwork(hidden_size=512).to(device)
    
    # Collect human demos. Demos will be cached for easier reuse.
    demo_file = './data/mountain_car_demos.pkl'
    if os.path.exists(demo_file):
        print(f"Found existing demos file: {demo_file}")
        demos = load_demos(demo_file)
        obs_demo, _, obs2_demo = torchify_demos(demos)
    else:
        print(f"No existing demos found. Collecting {num_demos} new demos...")
        demos = collect_human_demos(num_demos)
        save_demos(demos, demo_file)
        obs_demo, _, obs2_demo = torchify_demos(demos)
    
    # Line 3: Set I = number of samples to collect per BCO iteration
    I = num_interactions
    
    # s_s2_torch_ = None
    # a_torch_ = None
    
    # Line 4: `while policy improvement` do
    # NOTE: But I found it hard to directly implement `policy improvement`;
    # So I fallback to a fixed number of iterations, designated by `num_bco_iters`.
    # But doesn't matter here anyway since we only consider BCO(0) in the assignment.
    for iteration in range(num_bco_iters):
        print(f"BCO Iteration {iteration+1}/{num_bco_iters} (I={I})")
        
        # Lines 5-8: Collect samples from policy. Length is designated by `I`.
        s_s2_pi, acs_pi = collect_policy_interaction_data(pi, I)
        s_s2_torch = torch.from_numpy(s_s2_pi).float().to(device)
        a_torch = torch.from_numpy(acs_pi).long().to(device)
        # s_s2_torch_ = s_s2_torch if s_s2_torch_ is None else torch.cat((s_s2_torch_, s_s2_torch), dim=0)
        # a_torch_ = a_torch if a_torch_ is None else torch.cat((a_torch_, a_torch), dim=0)
        
        # Line 9: Improve M_theta by modelLearning
        train_inv_dyn(inv_dyn, s_s2_torch, a_torch, num_iters=200, learning_rate=1e-3)
        
        # Line 10-11: Get T_demo from human demonstrations; Then use M_theta with T_demo 
        # to approximate A_demo
        inv_dyn.eval()
        with torch.no_grad():
            T_demo = torch.cat((obs_demo, obs2_demo), dim=1)
            outputs = inv_dyn(T_demo)
            _, A_demo_pred = torch.max(outputs, 1)  # we only want the maximum one as predicted action
                
        # Line 12: Improve π_φ by behavioralCloning(S_demo, A_demo)
        train_policy(obs_demo, A_demo_pred, pi, num_bc_iters, batch_size=64)
        
        # Line 13: Set I = alpha * |I^pre| (for BCO(0), alpha=0, so I becomes 0 after first iteration)
        I = int(alpha * num_interactions)
        if I == 0:
            print(f"Stop iteration for BCO(0).")
            break
    
    torch.save(inv_dyn.state_dict(), 'data/mountain_car_inv_dyn.pt')
    torch.save(pi.state_dict(), 'data/mountain_car_policy_bco.pt')
    
    evaluate_policy(pi, num_evals)
    return


if __name__ == "__main__":
    # main_argparse()
    bco_main(num_demos=30, num_bc_iters=150, num_evals=10, alpha=0, num_bco_iters=10)