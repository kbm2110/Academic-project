import gymnasium as gym
import numpy as np
import torch
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.policy.base_policy import BasePolicy
from metadrive.utils.math import clip
from metadrive.engine.engine_utils import get_global_config


class PPOPolicy(BasePolicy):
    def __init__(self, obj, seed, actor_model, action_space, device):
        super(PPOPolicy, self).__init__(control_object=obj, random_seed=seed)
        self.actor_model = actor_model
        self.device = device
        self.action_space = action_space
        self.discrete_action = obj.engine.global_config["discrete_action"]
        self.use_multi_discrete = obj.engine.global_config["use_multi_discrete"]
        self.steering_unit = 2.0 / (obj.engine.global_config["discrete_steering_dim"] - 1)
        self.throttle_unit = 2.0 / (obj.engine.global_config["discrete_throttle_dim"] - 1)

    def act(self, agent_id):
        observation = self.control_object.get_state()

        if isinstance(observation, dict):
            observation = self._preprocess_observation(observation)

        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            dist = self.actor_model(obs_tensor)
            action = dist.sample().squeeze().cpu().numpy()

        action = [clip(action[i], -1.0, 1.0) for i in range(len(action))]

        self.action_info["action"] = action
        return action

    def _preprocess_observation(self, observation):

        processed_obs = []

        lidar_data = np.zeros(240)
        processed_obs.extend(lidar_data)

        if "position" in observation:
            processed_obs.extend(observation["position"])
        else:
            raise KeyError("Position key is missing in the observation.")

        if "velocity" in observation:
            processed_obs.extend(observation["velocity"])
        else:
            raise KeyError("Velocity key is missing in the observation.")

        processed_obs.append(observation.get("heading_theta", 0.0))
        processed_obs.append(observation.get("roll", 0.0))
        processed_obs.append(observation.get("pitch", 0.0))
        processed_obs.append(observation.get("steering", 0.0))
        processed_obs.append(observation.get("throttle_brake", 0.0))

        processed_obs.append(float(observation.get("crash_vehicle", False)))
        processed_obs.append(float(observation.get("crash_object", False)))
        processed_obs.append(float(observation.get("crash_building", False)))
        processed_obs.append(float(observation.get("crash_sidewalk", False)))

        processed_obs.extend(observation.get("size", [0.0, 0.0, 0.0]))

        # Add two additional scalar placeholders for consistency
        processed_obs.append(0.0)  # Placeholder 1
        processed_obs.append(0.0)  # Placeholder 2

        # Validate the processed observation size
        if len(processed_obs) != 259:
            raise ValueError(f"Processed observation size mismatch! Expected 259, got {len(processed_obs)}.")

        return np.array(processed_obs, dtype=np.float32)

    @classmethod
    def get_input_space(cls):

        engine_global_config = get_global_config()
        discrete_action = engine_global_config["discrete_action"]
        discrete_steering_dim = engine_global_config["discrete_steering_dim"]
        discrete_throttle_dim = engine_global_config["discrete_throttle_dim"]
        use_multi_discrete = engine_global_config["use_multi_discrete"]

        if not discrete_action:
            _input_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        else:
            if use_multi_discrete:
                _input_space = gym.spaces.MultiDiscrete([discrete_steering_dim, discrete_throttle_dim])
            else:
                _input_space = gym.spaces.Discrete(discrete_steering_dim * discrete_throttle_dim)
        return _input_space


def create_env_with_ppo_policy(actor_model, device):

    class MetaPPOPolicy(PPOPolicy):
        def __init__(self, obj, seed):
            action_space = MetaPPOPolicy.get_input_space()
            super().__init__(obj, seed, actor_model=actor_model, action_space=action_space, device=device)
    env_config = {
            "use_render": False,  # No rendering to speed up training
            "traffic_density": 0.2,  # Moderate traffic density for better challenges
            "random_traffic": True,  # Random traffic for variability
            "need_inverse_traffic": True,  # Reverse traffic for additional challenge
            "map": "S",  # Simple map for easy testing (could try others later)
            "vehicle_config": {
                "show_dest_mark": True,  # Show destination markers for goal-oriented behavior
                "show_navi_mark": True, # Show navigation markers for better guidance
                "lidar": {
                    "num_lasers": 240,
                    "distance": 75.0,
                    "gaussian_noise": 0.1,
                }  
            },
            "crash_object_done": True,  # Ends episode when crashes into object
            "crash_vehicle_done": True,  # Ends episode when vehicle crashes
            "out_of_route_done": True,  # Ends episode when off route
            "use_lateral_reward": True,  # Reward for maintaining lateral position
            "success_reward": 25.0,  # Moderate success reward for balanced learning
            "driving_reward": 2.0,  # Increased driving reward to encourage smooth driving
            "out_of_road_penalty": -8.0,  # High penalty for going out of road
            "crash_vehicle_penalty": -15.0,  # Large penalty for crashing into vehicles
            "crash_object_penalty": -8.0,  # Moderate penalty for crashing into objects
            "agent_observation": LidarStateObservation,
            "agent_policy": MetaPPOPolicy,  
        }

    # env_config = {
    #     "use_render": False,
    #     "traffic_density": 0.2,
    #     "need_inverse_traffic": True,
    #     "map": "S",
    #     "manual_control": False,
    #     "vehicle_config": {
    #         "lidar": {
    #             "num_lasers": 240,
    #             "distance": 50.0,
    #             "gaussian_noise": 0.1,
    #         },
    #         "show_lidar": False,
    #     },
    #     "success_reward": 20.0,
    #     "driving_reward": 2.0,
    #     "crash_vehicle_penalty": -10.0,
    #     "out_of_road_penalty": -5.0,
    #     "crash_vehicle_done": True,
    #     "out_of_road_done": True,
    #     "horizon": 1500,
    #     "agent_observation": LidarStateObservation,
    #     "agent_policy": MetaPPOPolicy,
    # }

    # Initialize MetaDrive environment
    env = MetaDriveEnv(config=env_config)

    def observation_processing(obs):

        try:
            if not isinstance(obs, np.ndarray):
                raise ValueError(f"Observation must be a numpy array, got {type(obs)}")
            expected_size = 259
            if obs.shape[0] != expected_size:
                raise ValueError(f"Observation size mismatch! Expected {expected_size}, got {obs.shape[0]}")
            return obs
        except Exception as e:
            print(f"Error in observation processing: {e}")
            return None

    env.process_observation = observation_processing
    return env


if __name__ == "__main__":
    class TestActor:
        def __init__(self):
            pass

        def __call__(self, x):
            return torch.distributions.Normal(torch.zeros(x.size(0), 2), torch.ones(x.size(0), 2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_model = TestActor() 

    env = create_env_with_ppo_policy(actor_model, device)
    obs, info = env.reset()

    processed_obs = env.process_observation(obs)
    if processed_obs is not None:
        print(f"Processed observation shape: {processed_obs.shape}")
        print(f" - Info: {info}\n")

    else:
        print("Observation processing failed.")

    env.close()