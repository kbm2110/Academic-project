import gymnasium as gym
from metadrive.envs.metadrive_env import MetaDriveEnv

def create_env():
    # Environment configuration parameters
    env_config = {
        "use_render": False,  # No rendering to speed up training
        "traffic_density": 0.2,  # Moderate traffic density for better challenges
        "random_traffic": True,  # Random traffic for variability
        "need_inverse_traffic": True,  # Reverse traffic for additional challenge
        "map": "S",  # Simple map for easy testing (could try others later)
        "vehicle_config": {
            "show_dest_mark": True,  # Show destination markers for goal-oriented behavior
            "show_navi_mark": True,  # Show navigation markers for better guidance
        },
        "crash_object_done": True,  # Ends episode when crashes into object
        "crash_vehicle_done": True,  # Ends episode when vehicle crashes
        "out_of_route_done": True,  # Ends episode when off route
        "use_lateral_reward": True,  # Reward for maintaining lateral position
        "success_reward": 10.0,  # Moderate success reward for balanced learning
        "driving_reward": 1.0,  # Increased driving reward to encourage smooth driving
        "out_of_road_penalty": -5.0,  # High penalty for going out of road
        "crash_vehicle_penalty": -5.0,  # Large penalty for crashing into vehicles
        "crash_object_penalty": -5.0,  # Moderate penalty for crashing into objects
    }

    """# Reward handling logic based on the 'info' dictionary
    def custom_reward_fn(info, reward):
        total_reward = reward
        
        # Apply penalties for crashes
        if info.get("crash_vehicle", False):
            total_reward += env_config["crash_vehicle_penalty"]
        elif info.get("crash_object", False):
            total_reward += env_config["crash_object_penalty"]
        
        # Apply penalty for going out of the road
        if info.get("out_of_road", False):
            total_reward += env_config["out_of_road_penalty"]
        
        # Reward for successfully arriving at the destination
        if info.get("arrive_dest", False):
            total_reward += env_config["success_reward"]
        
        # Driving reward for staying on the road (lateral control)
        if info.get("step_reward", 0) > 0:
            total_reward += env_config["driving_reward"]

        return total_reward"""

    # Create the MetaDrive environment and set the custom reward function
    env = MetaDriveEnv(config=env_config)
    #env.custom_reward_fn = custom_reward_fn  # Add custom reward calculation to the environment

    return env
