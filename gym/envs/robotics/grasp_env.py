import numpy as np

from gym.envs.robotics import rotations, robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class GraspEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments with camera input.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type, goal_type, cam_type,
        gripper_init_type, act_noise, obs_noise, joint_training
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            goal_type ('random' or 'fixed'): the goal type, i.e. random goal position or fixed goal position
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.goal_type = goal_type
        self.cam_type = cam_type
        self.gripper_init_type = gripper_init_type
        self.act_noise = act_noise
        self.obs_noise = obs_noise
        self.joint_training = joint_training

        if self.act_noise:
            noise_vector = np.random.uniform(-1.0, 1.0, 3)
            norm = np.linalg.norm(noise_vector)
            noise_vector_other = noise_vector / norm
            noise_vector = np.minimum(noise_vector, noise_vector_other)
            if norm == 0:
                self.act_noise_vector = np.zeros(3)
            else:
                if self.joint_training:
                    self.act_noise_vector = noise_vector * 0.002
                else:
                    self.act_noise_vector = noise_vector * 0.02
        else:
            self.act_noise_vector = np.zeros(3)

        if self.obs_noise:
            noise_vector = np.random.uniform(-1.0, 1.0, 3)
            norm = np.linalg.norm(noise_vector)
            noise_vector_other = noise_vector / norm
            noise_vector = np.minimum(noise_vector, noise_vector_other)
            if norm == 0:
                self.obs_noise_vector = np.zeros(3)
            else:
                if self.joint_training:
                    self.obs_noise_vector = noise_vector * 0.002
                else:
                    self.obs_noise_vector = noise_vector * 0.02
        else:
            self.obs_noise_vector = np.zeros(3)

        super(GraspEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if info == 'precise':
            return -(d > 0.01).astype(np.float32)
        elif self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        pos_ctrl += self.act_noise_vector # apply random noise
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

        # self.sim.step()


    def _get_obs(self):
        # images
        img = self.sim.render(width=500, height=500, camera_name="external_camera_1")
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        holder_pos = grip_pos.copy()
        grip_pos += self.obs_noise_vector
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = grip_pos
            object_rot = np.zeros(3)
            object_velp = grip_velp
            object_velr = np.zeros(3)
            object_rel_pos = np.zeros(3)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()# - self.sim.data.get_site_xpos("robot0:cam")
        else:
            achieved_goal = np.squeeze(object_pos.copy())# - self.sim.data.get_site_xpos("robot0:cam")
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
            'image':(img/255).copy(),
            'gripper_pose': holder_pos.copy()
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.goal_type == 'fixed':
            goal = self.initial_gripper_xpos[:3] + np.array([-0.15, 0.15, 0.15])
        elif self.has_object:
            goal = self.sim.data.get_site_xpos('object0').copy()
            goal[2] += 0.1
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()# - self.sim.data.get_site_xpos("robot0:cam")

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)

        if self.cam_type != "fixed":
            # delta_pos = self.np_random.uniform(-0.15, 0.15, size=3)
            delta_pos = np.array([self.np_random.uniform(0, 0.15), self.np_random.uniform(-0.1, 0.1), self.np_random.uniform(-0.1, 0.15)])
            delta_rot = self.np_random.uniform(-0.00, 0.00, size=3)
            utils.cam_init_pos(self.sim, delta_pos, delta_rot)

        self.sim.forward()

        # Move end effector into position.
        if self.gripper_init_type != "fixed":
            init_disturbance = np.array([self.np_random.uniform(-0.15, 0.15), self.np_random.uniform(-0.15, 0.15), self.np_random.uniform(0, 0.15)])
        else:
            init_disturbance = np.array([0, 0, 0])
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + init_disturbance + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='rgd_array', width=500, height=500):
        return super(GraspEnv, self).render(mode, width, height)