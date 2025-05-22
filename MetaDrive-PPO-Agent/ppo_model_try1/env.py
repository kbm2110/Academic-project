from metadrive.envs.metadrive_env import MetaDriveEnv

def create_env():
    """
    Create the MetaDrive environment configured for PPO training with custom termination conditions.
    """
    env_config = {
        "use_render": False,
        "traffic_density": 0.1,
        "need_inverse_traffic": True,
        "map": "S",
        "manual_control": False,
        "vehicle_config": {
            "show_dest_mark": True,
            "show_navi_mark": True,
        },
        "success_reward": 10.0,
        "driving_reward": 1.0,
        "horizon": 1500,
        "image_observation": False,
        "top_down_camera_initial_x": 100,
        "top_down_camera_initial_y": 100,
        "top_down_camera_initial_z": 120,
        "crash_vehicle_done": True,
        "crash_object_done": True,
        "out_of_route_done": True,  # Terminate the episode if the vehicle goes out of the route
    }

    env = MetaDriveEnv(config=env_config)
    return env

# Custom reward values for flexibility
reward_config = {
    "lane_penalty": -1.0,
    "crash_penalty": -10.0,
    "out_of_road_penalty": -5.0
}
