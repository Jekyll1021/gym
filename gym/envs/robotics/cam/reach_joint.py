import os
from gym import utils
from gym.envs.robotics import cam_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')


class CamReachJointEnv(cam_env.CamEnv, utils.EzPickle):
    def __init__(self, reward_type='dense', goal_type='fixed', cam_type='fixed', gripper_init_type='fixed', noise=False):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        cam_env.CamEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, goal_type=goal_type,
            cam_type=cam_type, gripper_init_type=gripper_init_type, noise=noise,
            joint_training=True)
        utils.EzPickle.__init__(self)
