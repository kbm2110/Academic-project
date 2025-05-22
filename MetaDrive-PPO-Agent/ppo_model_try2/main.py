import torch
import matplotlib.pyplot as plt
from env import create_env
from policy import PPOPolicy, PPOTrainer


exp_num = 1
def main():
    # Create environment
    env = create_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Initialize policy and trainer
    policy = PPOPolicy(obs_dim, act_dim)
    trainer = PPOTrainer(
        env=env,
        policy=policy,
        lr=1e-4,
        gamma=0.99,
        eps_clip=0.2
    )

    # Train the agent and capture rewards and losses
    rewards, losses = trainer.train(
        num_episodes=1000,
        steps_per_update=5000000,
        epochs=8
    )

    # Save the trained model
    model_path = f"ppo_metadrive_model_exp_num_{exp_num}.pth"
    torch.save(policy.state_dict(), model_path)
    print(f"Model saved at {model_path}")

    # Plot rewards over episodes
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards Over Episodes")
    plt.legend()
    plt.grid(True)

    # Plot loss over episodes
    plt.subplot(1, 2, 2)
    plt.plot(losses, label="Episode Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Episodes")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = f"training_rewards_and_loss_plot_exp_num_{exp_num}.png"
    plt.savefig(plot_path)
    print(f"Training plot saved at {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()
