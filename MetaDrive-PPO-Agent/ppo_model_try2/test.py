import torch
import matplotlib.pyplot as plt
from env import create_env
from policy import PPOPolicy

def test_model():
    # Create the environment using the existing configuration
    env = create_env()

    # Obtain observation and action space dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Load the pre-trained PPO policy model
    policy = PPOPolicy(obs_dim, act_dim)
    model_path = "ppo_metadrive_model_new.pth"
    policy.load_state_dict(torch.load(model_path))
    policy.eval()  # Set policy to evaluation mode

    # Parameters for evaluation
    num_episodes = 1000  # Number of evaluation episodes
    total_rewards = []

    # Evaluation loop
    for episode in range(num_episodes):
        obs, _ = env.reset()  # Reset the environment for each episode
        obs = torch.tensor(obs, dtype=torch.float32)
        episode_reward = 0
        done = False

        while not done:
            # Render the environment in top-down 2D mode
            env.render(mode="top_down")  # Ensures 2D rendering

            # Get action from the policy
            with torch.no_grad():
                action, _, _ = policy.get_action(obs)

            # Clip the action to the environment's bounds and execute it
            clipped_action = action.clamp(
                min=torch.tensor(env.action_space.low),
                max=torch.tensor(env.action_space.high)
            ).numpy()
            next_obs, reward, done, _, info = env.step(clipped_action)

            # Accumulate rewards and update observation
            episode_reward += reward
            obs = torch.tensor(next_obs, dtype=torch.float32)

        # Store episode reward
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}")

    # Close the environment
    env.close()

    # Plot the rewards across episodes
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_episodes + 1), total_rewards, marker='o', label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Evaluation Rewards Over Episodes")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test_model()
