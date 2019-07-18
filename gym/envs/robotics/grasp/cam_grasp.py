import os
from gym import utils
from gym.envs.robotics import grasp_env

# Ensure we get the path separator correct on windows
# MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')
MODEL_XML_PATH = os.path.join('fetch', 'random_obj_xml/000.xml')


class CamGraspEnv(grasp_env.GraspEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', goal_type='random', cam_type='fixed', gripper_init_type='fixed', act_noise=False, obs_noise=False, depth=False, two_cam=False, use_task_index=False, random_obj=False, train_random=False, test_random=False, limit_dir=False):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        grasp_env.GraspEnv.__init__(
            self, MODEL_XML_PATH, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.03, target_range=0.15, distance_threshold=0.03,
            initial_qpos=initial_qpos, reward_type=reward_type, goal_type=goal_type,
            cam_type=cam_type, gripper_init_type=gripper_init_type, act_noise=act_noise, obs_noise=obs_noise, depth=depth, two_cam=two_cam, use_task_index=use_task_index, random_obj=random_obj, train_random=train_random, test_random=test_random, limit_dir=limit_dir)
        utils.EzPickle.__init__(self)
