# def get_image_from_maniskill2_obs_dict(env, obs, camera_name=None):
#     # obtain image from observation dictionary returned by ManiSkill2 environment
    
#     if camera_name is None:
#         if "google_robot" in env.robot_uid:
#             camera_name = "overhead_camera"
#         elif "widowx" in env.robot_uid:
#             camera_name = "3rd_view_camera"
#         else:
#             raise NotImplementedError()
#     return obs["image"][camera_name]["rgb"]
def get_image_from_maniskill2_obs_dict(env, obs, camera_name=None):
    # If the environment is wrapped (e.g., with TimeLimit), get the underlying environment.
    if hasattr(env, "unwrapped"):
        env = env.unwrapped

    if camera_name is None:
        # Check for 'robot_uid' and set the camera accordingly.
        if hasattr(env, "robot_uid") and "google_robot" in env.robot_uid:
            camera_name = "overhead_camera"
        elif hasattr(env, "robot_uid") and "widowx" in env.robot_uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError("Unknown robot type or 'robot_uid' not found.")
    return obs["image"][camera_name]["rgb"]
