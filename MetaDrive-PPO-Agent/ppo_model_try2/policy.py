import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class PPOPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super(PPOPolicy, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )

        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, obs):
        mean_action = self.policy_net(obs)
        state_value = self.value_net(obs)
        return mean_action, state_value

    def get_action(self, obs):
        mean_action, _ = self.forward(obs)
        std = self.log_std.exp()
        dist = Normal(mean_action, std)
        action = dist.sample()
        return action, dist.log_prob(action).sum(), dist.entropy().sum()


class PPOTrainer:
    def __init__(self, env, policy, lr, gamma, eps_clip):
        self.env = env
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.reward_config = env.config  # Access the reward configuration directly from the environment's config

    def compute_metadrive_reward(self, reward, info):
        # Account for all the custom rewards and penalties from the environment config
        total_reward = reward
        
        # Apply penalties for crashes
        if info.get("crash_vehicle", False):
            total_reward += self.reward_config["crash_vehicle_penalty"]
        elif info.get("crash_object", False):
            total_reward += self.reward_config["crash_object_penalty"]
        
        # Apply penalty for going out of the road
        if info.get("out_of_road", False):
            total_reward += self.reward_config["out_of_road_penalty"]
        
        # Reward for successfully arriving at the destination
        if info.get("arrive_dest", False):
            total_reward += self.reward_config["arrive_dest_reward"]
        
        # Driving reward for staying on the road (lateral control)
        if info.get("step_reward", 0) > 0:
            total_reward += self.reward_config["driving_reward"]

        return total_reward

    def train(self, num_episodes, steps_per_update, epochs):
        all_rewards = []
        all_losses = []
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            obs = torch.tensor(obs, dtype=torch.float32)
            states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

            for step in range(steps_per_update):
                action, log_prob, _ = self.policy.get_action(obs)
                clipped_action = action.clamp(
                    min=self.env.action_space.low[0], max=self.env.action_space.high[0]
                )
                next_obs, reward, done, truncated, info = self.env.step(clipped_action.numpy())
                custom_reward = self.compute_metadrive_reward(reward, info)

                states.append(obs)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(custom_reward)
                dones.append(done)
                values.append(self.policy.value_net(obs).item())

                obs = torch.tensor(next_obs, dtype=torch.float32)

                # Handle termination and truncation conditions based on the provided flags in info
                if done or truncated:
                    break

            values.append(self.policy.value_net(obs).item())
            advantages = self._compute_advantages(rewards, values, dones)
            returns = advantages + torch.tensor(values[:-1], dtype=torch.float32)

            episode_loss = 0
            for _ in range(epochs):
                log_probs = torch.tensor(log_probs, dtype=torch.float32)
                loss = self._ppo_update(states, actions, log_probs, advantages, returns)
                episode_loss += loss.item()

            all_rewards.append(sum(rewards))
            all_losses.append(episode_loss / epochs)
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {sum(rewards):.2f}, Loss: {episode_loss:.2f}")

        return all_rewards, all_losses

    def _compute_advantages(self, rewards, values, dones):
        advantages = []
        advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (values[t + 1] if t + 1 < len(values) else 0) - values[t]
            advantage = delta + self.gamma * advantage
            advantages.insert(0, advantage)
        return torch.tensor(advantages, dtype=torch.float32)

    def _ppo_update(self, states, actions, log_probs, advantages, returns):
        new_log_probs = []
        entropy = []
        for i, state in enumerate(states):
            mean_action, _ = self.policy(state)
            std = self.policy.log_std.exp()
            dist = Normal(mean_action, std)
            new_log_probs.append(dist.log_prob(actions[i]).sum())
            entropy.append(dist.entropy().sum())

        new_log_probs = torch.stack(new_log_probs)
        ratio = (new_log_probs - log_probs.detach()).exp()
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        value_loss = nn.MSELoss()(self.policy.value_net(torch.stack(states)).squeeze(-1), returns)

        loss = policy_loss + 0.5 * value_loss - 0.01 * torch.stack(entropy).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
