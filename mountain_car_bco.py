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

# Use cuda for faster training if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collect_random_interaction_data(num_iters: int) -> tuple[np.ndarray, np.ndarray]:
    """Collect random state transitions from the environment. Will be later used to 
    initialize inverse dynamics model with random data **before** BCO loop.

    As I understand it, this can help the inverse dynamics model have 
    initial knowledge about the environment dynamics, rather than starting
    as a completely random approximator.
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


def collect_policy_interaction_data(policy: PolicyNetwork, num_iters: int, epsilon: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect state transitions using current policy network with epsilon-greedy exploration
    Corresponds to line 6, 7 in BCO algorithm.
    For BCO(0), this is only done in the first iteration of the BCO loop.
    """
    state_next_state = []
    actions = []
    env = gym.make('MountainCar-v0')
    
    policy.eval()
    for _ in range(num_iters):
        obs = env.reset()
        done = False
        
        while not done:
            # Epsilon-greedy exploration for better state coverage
            if np.random.random() < epsilon:
                a = env.action_space.sample()
            else:
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
        Neural network that maps (s,s') state to a prediction
        over which of the three discrete actions was taken.
        The network should have three outputs corresponding to the logits for a 3-way classification problem.

    '''
    def __init__(self, hidden_size=512):
        super().__init__()

        # Deeper network for better inverse dynamics modeling
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
        # Forward pass with batch normalization
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x
    

def train_inv_dyn(inv_dyn: InvDynamicsNetwork, s_s2_torch: torch.Tensor, 
                  a_torch: torch.Tensor, num_iters: int=100,
                  learning_rate: float=1e-3, batch_size: int=128):
    optimizer = Adam(inv_dyn.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    inv_dyn.train()
    dataset_size = s_s2_torch.shape[0]
    
    for epoch in range(num_iters):
        indices = torch.randperm(dataset_size)
        total_loss = 0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, dataset_size, batch_size):
            batch_indices = indices[i:min(i + batch_size, dataset_size)]
            batch_s_s2 = s_s2_torch[batch_indices]
            batch_a = a_torch[batch_indices]
            
            optimizer.zero_grad()
            logits = inv_dyn(batch_s_s2)
            loss = F.cross_entropy(logits, batch_a)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(inv_dyn.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 50 == 0:
            print(f"  InvDyn Epoch {epoch + 1}/{num_iters}, Loss: {avg_loss:.4f}")
    
    inv_dyn.eval()
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
         alpha: float = 0.0, num_bco_iters: int = 10):
    """
    I reformatted the code to match the lines in the BCO algorithm pseudocode.
    """
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Initial GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # Line 1: Initialize the model M_θ as random approximator
    inv_dyn = InvDynamicsNetwork(hidden_size=512).to(device)
    
    # Train M_θ with initial random data but related to the game environment
    num_interactions = 300  # Increased for better initialization
    print(f"\nCollecting {num_interactions} random interactions for inverse dynamics initialization...")
    s_s2, acs = collect_random_interaction_data(num_interactions)
    s_s2_torch = torch.from_numpy(np.array(s_s2)).float().to(device)
    a_torch = torch.from_numpy(np.array(acs)).long().to(device)
    print("Training initial inverse dynamics model...")
    train_inv_dyn(inv_dyn, s_s2_torch, a_torch, num_iters=200, learning_rate=1e-3)
    
    pi = PolicyNetwork(hidden_size=256).to(device)
    
    # Collect human demos once (Line 10: demonstrated state trajectories D_demo)
    demo_file = 'mountain_car_demos.pkl'
    if os.path.exists(demo_file):
        print(f"\nFound existing demos file: {demo_file}")
        demos = load_demos(demo_file)
        obs_demo, _, obs2_demo = torchify_demos(demos)
    else:
        print(f"\nNo existing demos found. Collecting {num_demos} new demos...")
        demos = collect_human_demos(num_demos)
        save_demos(demos, demo_file)
        obs_demo, _, obs2_demo = torchify_demos(demos)
    
    print(f"Collected {len(obs_demo)} demonstration transitions\n")
    
    # Line 3: Set I = number of episodes to collect per iteration
    I = num_interactions
    
    # fallback `policy improvement` to a fixed number of iterations
    s_s2_torch_ = None
    a_torch_ = None
    
    print("=" * 60)
    for iteration in range(num_bco_iters):
        print(f"\n--- BCO Iteration {iteration+1}/{num_bco_iters} (I={I}) ---")
        if torch.cuda.is_available():
            print(f"GPU memory usage: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        
        # Lines 5-8: for time-step t=1 to I do
        print(f"\n--- BCO Iteration {iteration+1}/{num_bco_iters} (I={I}) ---")
        
        # Lines 5-8: for time-step t=1 to I do
        # Line 6-7: Generate samples and append to datasets using π_φ
        epsilon = max(0.05, 0.2 - iteration * 0.015)  # Decay exploration
        print(f"Collecting policy interactions (epsilon={epsilon:.3f})...")
        s_s2_pi, acs_pi = collect_policy_interaction_data(pi, I, epsilon=epsilon)
        s_s2_torch = torch.from_numpy(s_s2_pi).float().to(device)
        a_torch = torch.from_numpy(acs_pi).long().to(device)
        s_s2_torch_ = s_s2_torch if s_s2_torch_ is None else torch.cat((s_s2_torch_, s_s2_torch), dim=0)
        a_torch_ = a_torch if a_torch_ is None else torch.cat((a_torch_, a_torch), dim=0)
        
        # Line 9: Improve M_θ by modelLearning
        print(f"Training inverse dynamics model on {len(s_s2_torch_)} transitions...")
        train_inv_dyn(inv_dyn, s_s2_torch_, a_torch_, num_iters=300, learning_rate=5e-4)
        
        # Line 10-11: Use M_θ with T_demo to approximate A_demo
        print("Predicting demonstration actions with inverse dynamics...")
        inv_dyn.eval()
        with torch.no_grad():
            state_trans_demo = torch.cat((obs_demo, obs2_demo), dim=1)
            outputs = inv_dyn(state_trans_demo)
            _, acs_pred = torch.max(outputs, 1)
        
        # Calculate prediction accuracy
        with torch.no_grad():
            pred_probs = F.softmax(outputs, dim=1)
            max_probs = pred_probs.max(dim=1)[0]
            avg_confidence = max_probs.mean().item()
        print(f"Inverse dynamics prediction confidence: {avg_confidence:.3f}")
        
        # Line 12: Improve π_φ by behavioralCloning(S_demo, A_demo)
        train_policy(obs_demo, acs_pred, pi, num_bc_iters, batch_size=64)
        
        # Intermediate evaluation
        if (iteration + 1) % 3 == 0:
            print(f"\nIntermediate evaluation at iteration {iteration + 1}:")
            evaluate_policy(pi, num_evals=3)
        
        # Line 13: Set I = α|I^pre| (for BCO(0), α=0, so I becomes 0 after first iteration)
        I = int(alpha * num_interactions)
        if I == 0:
            print(f"\nI = 0 (BCO(0)), stopping after this iteration")
            break
    
    # Save trained models
    torch.save(inv_dyn.state_dict(), 'mountain_car_inv_dyn.pt')
    torch.save(pi.state_dict(), 'mountain_car_policy_bco.pt')
    print("\n" + "=" * 60)
    print("Models saved: mountain_car_inv_dyn.pt, mountain_car_policy_bco.pt")
    print("=" * 60)
    
    # Evaluate learned policy
    print("\n=== Final Evaluation ===")
    evaluate_policy(pi, num_evals)


if __name__ == "__main__":
    # main_argparse()
    main(num_demos=15, num_bc_iters=150, num_evals=10, alpha=0, num_bco_iters=10)