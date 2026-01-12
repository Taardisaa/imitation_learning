import numpy as np
# Fix for numpy compatibility issue - add bool8 alias if it doesn't exist
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# Import OpenAI Gym library for reinforcement learning environments
import gym

# Create the MountainCar environment
# MountainCar-v0: A car must drive up a steep hill, but lacks power to climb directly
env = gym.make('MountainCar-v0')

env.reset()
done = False
while not done:
    # Take a random action from the action space and get the result
    # Returns: observation (state), reward, done (terminal flag), info (additional data)
    sampled_data = env.action_space.sample()
    observation, reward, done, info = env.step(sampled_data)
    print(observation, reward, done, info)
    # Render the environment to visualize the car's position
    env.render()

# Close the environment and clean up resources
env.close()
