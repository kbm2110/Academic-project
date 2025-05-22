import torch
import numpy as np
import matplotlib.pyplot as plt
from main import create_env_with_ppo_policy
from ppo import Actor, Critic

# Load the trained models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

actor = Actor(input_dim=259, action_dim=2).to(device)
critic = Critic(input_dim=259).to(device)

actor.load_state_dict(torch.load("ppo_actor.pth"))
critic.load_state_dict(torch.load("ppo_critic.pth"))

# Set the models to evaluation mode
actor.eval()
critic.eval()

# Create environment
env = create_env_with_ppo_policy(actor, device)

# Parameters for testing
num_episodes = 10  # Set the number of episodes you want to run
max_steps_per_episode = 500  # Set maximum steps per episode

# Variables to track results
episode_rewards = []  # Store rewards for each episode
episode_lengths = []  # Store lengths of each episode (steps before termination)

# Run the environment for multiple episodes
for episode in range(num_episodes):
    obs, _ = env.reset()
    processed_obs = env.process_observation(obs)

    total_reward = 0
    trajectory_obs = []
    trajectory_acts = []

    for step in range(max_steps_per_episode):
        obs_tensor = torch.tensor(processed_obs, dtype=torch.float32, device=device).unsqueeze(0)
        dist = actor(obs_tensor)
        action = dist.sample()
        action = action.squeeze().cpu().numpy()

        # Take a step in the environment
        next_raw_obs, reward, terminated, truncated, info = env.step(action)
        processed_obs = env.process_observation(next_raw_obs)

        # Collect data for visualization
        trajectory_obs.append(processed_obs)
        trajectory_acts.append(action)
        total_reward += reward

        env.render(mode="topdown")
        #print(info)
        if terminated or truncated:
            break

    # Record the results for the current episode
    episode_rewards.append(total_reward)
    episode_lengths.append(len(trajectory_obs))

    print(f"Episode {episode+1}: Total Reward = {total_reward}, Steps = {len(trajectory_obs)}")

# Visualize the results over episodes
plt.figure(figsize=(12, 6))

# Plot rewards per episode
plt.subplot(1, 2, 1)
plt.plot(range(1, num_episodes + 1), episode_rewards, marker='o', color='b', label='Total Reward')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Rewards per Episode")
plt.legend()

# Plot episode lengths (steps before termination)
plt.subplot(1, 2, 2)
plt.plot(range(1, num_episodes + 1), episode_lengths, marker='o', color='g', label='Episode Length')
plt.xlabel("Episode")
plt.ylabel("Steps (Episode Length)")
plt.title("Episode Lengths (Steps before Termination)")
plt.legend()

plt.tight_layout()
plt.show()

env.close()
