import os
import numpy as np

from gym import utils
from gym.envs.robotics import cam_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'slide.xml')


class CamSlideEnv(cam_env.CamEnv, utils.EzPickle):
    def __init__(self, reward_type='dense', goal_type='fixed', cam_type='fixed', gripper_init_type='fixed', act_noise=False, obs_noise=False):
        initial_qpos = {
            'robot0:slide0': 0.05,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.7, 1.1, 0.4, 1., 0., 0., 0.],
        }
        cam_env.CamEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=-0.02, target_in_the_air=False, target_offset=np.array([0.4, 0.0, 0.0]),
            obj_range=0.1, target_range=0.3, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, goal_type=goal_type,
            cam_type=cam_type, gripper_init_type=gripper_init_type, act_noise=act_noise, obs_noise=obs_noise,
            joint_training=False)
        utils.EzPickle.__init__(self)
