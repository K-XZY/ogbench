from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from dm_control import mjcf
from gymnasium.spaces import Box

from ogbench.manipspace import controllers, lie, mjcf_utils
from ogbench.manipspace.envs.env import CustomMuJoCoEnv


# Default pose of the optional `wrist` camera, expressed in the Robotiq 2F85 `base` frame.
# In that frame +z is the gripper approach axis (the `pinch` site sits at (0, 0, 0.145)) and the
# fingers open along +-y. The camera sits behind and to the side of the pinch and looks at it, so
# the fingers land left/right in the image and a grasped cube is centered. Keep these as config
# values: they are derived from the MJCF geometry and are expected to be tuned from a render.
DEFAULT_WRIST_CAMERA = dict(
    pos=(-0.06, 0.0, 0.03),
    xyaxes=(0.0, 1.0, 0.0, 0.8866, 0.0, -0.4626),
    fovy=65.0,
)

# Default pose of the optional `overhead` camera, in world coordinates. Top-down over the workspace
# centre, looking straight down (-z). Image-right is world +y, matching the `front` cameras, so
# image-up is world -x, i.e. toward the robot base. At 0.68 m above the table fovy=60 covers roughly
# y in [-0.39, 0.39] and x in [0.03, 0.82], which holds the object sampling bounds with margin.
# Config values, not literals: expected to be tuned from a render.
DEFAULT_OVERHEAD_CAMERA = dict(
    pos=(0.425, 0.0, 0.7),
    xyaxes=(0.0, 1.0, 0.0, -1.0, 0.0, 0.0),
    fovy=60.0,
)

# Optional extra lighting for the VLA render path.
#
# Upstream is dim -- `front` averages ~46/255, about 18% of range, where natural photographs a
# pretrained encoder was trained on sit nearer 45-50%. Note that the light-removal loop in
# `build_mjcf_model` only strips lights from the *UR5e* MJCF (a single spotlight targeting
# `wrist_2_link`); the arena keeps its `global` directional light and its `spotlight`. So this is
# not restoring deleted lights, it is adding exposure on top of what already exists.
#
# `ambient` does the heavy lifting: it lifts shadowed regions, where the detail is lost, without
# pushing already-lit metal toward clipping. The added area light over the workspace fills the
# table. `castshadow` stays off so we do not introduce a second shadow that conflicts with the
# existing ones.
DEFAULT_RENDER_LIGHTING = dict(
    headlight_diffuse=0.8,
    headlight_ambient=0.35,
    workspace_light=dict(
        pos=(0.425, 0.0, 1.0),
        dir=(0.0, 0.0, -1.0),
        diffuse=(0.4, 0.4, 0.4),
        specular=(0.0, 0.0, 0.0),
        cutoff=70.0,
        castshadow=False,
    ),
)

# Friendly camera names -> MJCF identifiers. The dataset records `front`/`overhead`/`wrist`; the
# wrist camera is mounted inside the gripper's attachment namespace, so its compiled name is
# prefixed, and OGBench's pixel camera is called `front_pixels`.
CAMERA_ALIASES = {
    'front': 'front_pixels',
    'overhead': 'overhead',
    'wrist': 'ur5e/robotiq/wrist',
}

# `visual/map znear` is a *fraction of* `statistic.extent`, so the upstream 0.1 with extent 0.7
# puts the near plane at 0.07 m -- which would slice off the proximal half of the fingers in the
# wrist view. 0.02 puts it at 0.014 m. Only depth precision suffers, and we render RGB only.
DEFAULT_RENDER_ZNEAR = 0.02


class ManipSpaceEnv(CustomMuJoCoEnv):
    """ManipSpace environment.

    This is the base class for all OGBench manipulation task. It contains a UR5e robot arm with a Robotiq 2F-85 gripper.
    The default control mode is relative end-effector control. The 5-D action space corresponds to the following:
    - 3-D relative end-effector position (x, y, z).
    - 1-D relative end-effector yaw.
    - 1-D relative gripper opening.
    """

    def __init__(
        self,
        ob_type='states',
        physics_timestep=0.002,
        control_timestep=0.05,
        terminate_at_goal=True,
        success_timing='post',
        mode='task',
        visualize_info=True,
        pixel_transparent_arm=True,
        pixel_recolor_arm=True,
        render_camera_names=None,
        wrist_camera=None,
        overhead_camera=None,
        visual_znear=None,
        render_lighting=None,
        reward_task_id=None,
        use_oracle_rep=False,
        **kwargs,
    ):
        """Initialize the ManipSpace environment.

        Args:
            ob_type: Observation type. Either 'states' or 'pixels'.
            physics_timestep: Physics timestep.
            control_timestep: Control timestep.
            terminate_at_goal: Whether to terminate the episode when the goal is reached.
            success_timing: When to compute the success and terminated flags and the reward. Either 'pre' (before taking
                the action) or 'post' (after taking the action).
            mode: Mode of the environment. Either 'task' or 'data_collection'. In 'task' mode, the environment is used
                for training and evaluation. In 'data_collection' mode, the environment is used for collecting offline
                data.
            visualize_info: Whether to visualize the task information (e.g., success status).
            pixel_transparent_arm: Whether to make the arm transparent in pixel-based observations.
            pixel_recolor_arm: Whether to recolor the gripper purple in pixel-based observations. Set to False for
                realistic appearance.
            render_camera_names: If not None, a list of camera names that `render_cameras` renders, enabling the
                multi-camera render path. Because that path exists to produce training observations, it refuses to
                run with `visualize_info=True`, which would paint the goal into the image.
            wrist_camera: Wrist camera configuration. None disables it (upstream behavior); True uses
                `DEFAULT_WRIST_CAMERA`; a dict overrides individual `pos`/`xyaxes`/`fovy` entries of that default.
            overhead_camera: Overhead camera configuration, same convention as `wrist_camera` but relative to
                `DEFAULT_OVERHEAD_CAMERA`.
            visual_znear: Near clipping plane as a fraction of `statistic.extent`. None keeps the upstream 0.1;
                the wrist camera needs `DEFAULT_RENDER_ZNEAR`.
            render_lighting: Extra lighting for the render path. None keeps upstream lighting; True uses
                `DEFAULT_RENDER_LIGHTING`; a dict overrides its individual entries. Recorded in the dataset config,
                since it changes what every frame looks like.
            reward_task_id: Task ID for single-task RL. If this is not None, the environment operates in a single-task
            mode with the specified task ID. The task ID must be either a valid task ID or 0, where 0 means using the
            default task.
            use_oracle_rep: Whether to use oracle goal representations.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            physics_timestep=physics_timestep,
            control_timestep=control_timestep,
            **kwargs,
        )

        # Define constants.
        self._desc_dir = Path(__file__).resolve().parent / '..' / 'descriptions'
        self._home_qpos = np.asarray([-np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])
        self._effector_down_rotation = lie.SO3(np.asarray([0.0, 1.0, 0.0, 0.0]))
        self._workspace_bounds = np.asarray([[0.25, -0.35, 0.02], [0.6, 0.35, 0.35]])
        self._arm_sampling_bounds = np.asarray([[0.25, -0.35, 0.20], [0.6, 0.35, 0.35]])
        self._object_sampling_bounds = np.asarray([[0.3, -0.3], [0.55, 0.3]])
        self._target_sampling_bounds = np.asarray([[0.3, -0.3], [0.55, 0.3]])
        self._colors = dict(
            red=np.array([0.96, 0.26, 0.33, 1.0]),
            orange=np.array([1.0, 0.69, 0.21, 1.0]),
            yellow=np.array([0.76, 0.96, 0.04, 1.0]),
            green=np.array([0.06, 0.74, 0.21, 1.0]),
            blue=np.array([0.35, 0.55, 0.91, 1.0]),
            purple=np.array([0.61, 0.28, 0.82, 1.0]),
            magenta=np.array([0.82, 0.28, 0.61, 1.0]),
            lightred=np.array([0.99, 0.85, 0.86, 1.0]),
            lightorange=np.array([1.0, 0.94, 0.84, 1.0]),
            lightyellow=np.array([0.95, 0.99, 0.8, 1.0]),
            lightgreen=np.array([0.77, 0.95, 0.81, 1.0]),
            lightblue=np.array([0.86, 0.9, 0.98, 1.0]),
            lightpurple=np.array([0.91, 0.84, 0.96, 1.0]),
            lightmagenta=np.array([0.96, 0.84, 0.91, 1.0]),
            white=np.array([0.9, 0.9, 0.9, 1.0]),
            lightgray=np.array([0.7, 0.7, 0.7, 1.0]),
            gray=np.array([0.5, 0.5, 0.5, 1.0]),
            darkgray=np.array([0.3, 0.3, 0.3, 1.0]),
            black=np.array([0.1, 0.1, 0.1, 1.0]),
        )

        self._ob_type = ob_type
        self._terminate_at_goal = terminate_at_goal
        self._success_timing = success_timing
        self._mode = mode
        self._visualize_info = visualize_info
        self._pixel_transparent_arm = pixel_transparent_arm
        self._pixel_recolor_arm = pixel_recolor_arm
        self._reward_task_id = reward_task_id
        self._use_oracle_rep = use_oracle_rep

        # Wrist camera configuration.
        if wrist_camera is None or wrist_camera is False:
            self._wrist_camera = None
        elif wrist_camera is True:
            self._wrist_camera = dict(DEFAULT_WRIST_CAMERA)
        else:
            self._wrist_camera = {**DEFAULT_WRIST_CAMERA, **wrist_camera}

        if overhead_camera is None or overhead_camera is False:
            self._overhead_camera = None
        elif overhead_camera is True:
            self._overhead_camera = dict(DEFAULT_OVERHEAD_CAMERA)
        else:
            self._overhead_camera = {**DEFAULT_OVERHEAD_CAMERA, **overhead_camera}

        if render_lighting is None or render_lighting is False:
            self._render_lighting = None
        elif render_lighting is True:
            self._render_lighting = dict(DEFAULT_RENDER_LIGHTING)
        else:
            self._render_lighting = {**DEFAULT_RENDER_LIGHTING, **render_lighting}

        self._visual_znear = visual_znear
        self._render_camera_names = None if render_camera_names is None else list(render_camera_names)

        # Segment bookkeeping for data collection. A segment begins at every `set_new_target` call.
        self._segment_index = -1
        self._segment_step = 0

        assert ob_type in ['states', 'pixels']
        assert success_timing in ['pre', 'post']
        if self._render_camera_names is not None and visualize_info:
            # `visualize_info` draws the target cube as a translucent ghost and recolors cubes on success, both of
            # which put goal and reward information directly into the pixels. Never render training data with it on.
            raise ValueError(
                'render_camera_names requires visualize_info=False; otherwise the goal leaks into the image.'
            )

        # Initialize inverse kinematics controller.
        ik_mjcf = mjcf.from_path((self._desc_dir / 'universal_robots_ur5e' / 'ur5e.xml'), escape_separators=True)
        xml_str = mjcf_utils.to_string(ik_mjcf)
        assets = mjcf_utils.get_assets(ik_mjcf)
        ik_model = mujoco.MjModel.from_xml_string(xml_str, assets)

        self._ik = controllers.DiffIKController(model=ik_model, sites=['attachment_site'])

        # Define action space.
        action_range = np.array([0.05, 0.05, 0.05, 0.3, 1.0])
        self.action_low = -action_range
        self.action_high = action_range

        if self._mode == 'task':
            # Set task goals.
            self.task_infos = []
            self.cur_task_id = None
            self.cur_task_info = None
            self.set_tasks()
            self.num_tasks = len(self.task_infos)

            self._cur_goal_ob = None
            self._cur_goal_rendered = None
            self._render_goal = False

        self._success = False

    @property
    def observation_space(self):
        if self._model is None:
            self.reset()

        ex_ob = self.compute_observation()

        if self._ob_type == 'pixels':
            return Box(low=0, high=255, shape=ex_ob.shape, dtype=ex_ob.dtype)
        else:
            return Box(low=-np.inf, high=np.inf, shape=ex_ob.shape, dtype=ex_ob.dtype)

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=-np.ones(5),
            high=np.ones(5),
            shape=(5,),
            dtype=np.float32,
        )

    def normalize_action(self, action):
        """Normalize the action to the range [-1, 1]."""
        action = 2 * (action - self.action_low) / (self.action_high - self.action_low) - 1
        return np.clip(action, -1, 1)

    def unnormalize_action(self, action):
        """Unnormalize the action to the range [action_low, action_high]."""
        return 0.5 * (action + 1) * (self.action_high - self.action_low) + self.action_low

    def set_tasks(self):
        pass

    def build_mjcf_model(self):
        # Set scene.
        arena_mjcf = mjcf.from_path((self._desc_dir / 'floor_wall.xml').as_posix())
        arena_mjcf.model = 'ur5e_arena'

        arena_mjcf.statistic.center = (0.3, 0, 0.15)
        arena_mjcf.statistic.extent = 0.7
        getattr(arena_mjcf.visual, 'global').elevation = -20
        getattr(arena_mjcf.visual, 'global').azimuth = 180
        arena_mjcf.statistic.meansize = 0.04
        arena_mjcf.visual.map.znear = 0.1 if self._visual_znear is None else self._visual_znear
        arena_mjcf.visual.map.zfar = 10.0

        # Add UR5e robot arm.
        ur5e_mjcf = mjcf.from_path((self._desc_dir / 'universal_robots_ur5e' / 'ur5e.xml'), escape_separators=True)
        ur5e_mjcf.model = 'ur5e'

        for light in ur5e_mjcf.find_all('light'):
            light.remove()
            del light

        # Attach the robotiq gripper to the UR5e flange.
        gripper_mjcf = mjcf.from_path((self._desc_dir / 'robotiq_2f85' / '2f85.xml'), escape_separators=True)
        gripper_mjcf.model = 'robotiq'

        if self._wrist_camera is not None:
            # Mount on the gripper `base` body, so the camera moves with the hand.
            gripper_mjcf.find('body', 'base').add('camera', name='wrist', **self._wrist_camera)

        mjcf_utils.attach(ur5e_mjcf, gripper_mjcf, 'attachment_site')

        # Attach UR5e to the scene.
        mjcf_utils.attach(arena_mjcf, ur5e_mjcf)

        if self._render_lighting is not None:
            lighting = dict(self._render_lighting)
            workspace_light = lighting.pop('workspace_light', None)
            headlight_diffuse = lighting.pop('headlight_diffuse', None)
            headlight_ambient = lighting.pop('headlight_ambient', None)
            if lighting:
                raise ValueError(f'Unknown render_lighting keys: {sorted(lighting)}')

            if headlight_diffuse is not None:
                arena_mjcf.visual.headlight.diffuse = (headlight_diffuse,) * 3
            if headlight_ambient is not None:
                arena_mjcf.visual.headlight.ambient = (headlight_ambient,) * 3
            if workspace_light is not None:
                arena_mjcf.worldbody.add('light', name='workspace', **workspace_light)

        if self._overhead_camera is not None:
            arena_mjcf.worldbody.add('camera', name='overhead', **self._overhead_camera)

        self.add_objects(arena_mjcf)

        # Cache joint and actuator elements.
        self._arm_jnts = mjcf_utils.safe_find_all(
            ur5e_mjcf,
            'joint',
            exclude_attachments=True,
        )
        self._arm_acts = mjcf_utils.safe_find_all(
            ur5e_mjcf,
            'actuator',
            exclude_attachments=True,
        )
        self._gripper_jnts = mjcf_utils.safe_find_all(gripper_mjcf, 'joint', exclude_attachments=True)
        self._gripper_acts = mjcf_utils.safe_find_all(gripper_mjcf, 'actuator', exclude_attachments=True)

        if self._ob_type == 'pixels':
            # Adjust colors for pixel-based tasks.
            if self._pixel_recolor_arm:
                arena_mjcf.find('material', 'ur5e/robotiq/black').rgba = self._colors['purple']
                arena_mjcf.find('material', 'ur5e/robotiq/pad_gray').rgba = self._colors['purple']
            if self._pixel_transparent_arm:
                arena_mjcf.find('material', 'ur5e/robotiq/metal').rgba[3] = 0.1
                arena_mjcf.find('material', 'ur5e/robotiq/silicone').rgba[3] = 0.1
                arena_mjcf.find('material', 'ur5e/robotiq/gray').rgba[3] = 0.1
                arena_mjcf.find('material', 'ur5e/robotiq/black').rgba[3] = 0.1
                arena_mjcf.find('material', 'ur5e/robotiq/pad_gray').rgba[3] = 0.5
                arena_mjcf.find('material', 'ur5e/black').rgba[3] = 0.1
                arena_mjcf.find('material', 'ur5e/jointgray').rgba[3] = 0.1
                arena_mjcf.find('material', 'ur5e/linkgray').rgba[3] = 0.1
                arena_mjcf.find('material', 'ur5e/lightblue').rgba[3] = 0.1

        # Add bounding boxes to visualize the workspace and object sampling bounds.
        mjcf_utils.add_bounding_box_site(
            arena_mjcf.worldbody,
            lower=np.asarray((*self._target_sampling_bounds[0], 0.02)),
            upper=np.asarray((*self._target_sampling_bounds[1], 0.02)),
            rgba=(0.6, 0.3, 0.3, 0.2),
            group=4,
            name='object_bounds',
        )
        mjcf_utils.add_bounding_box_site(
            arena_mjcf.worldbody,
            lower=np.asarray(self._arm_sampling_bounds[0]),
            upper=np.asarray(self._arm_sampling_bounds[1]),
            rgba=(0.3, 0.6, 0.3, 0.2),
            group=4,
            name='arm_bounds',
        )

        return arena_mjcf

    def add_objects(self, arena_mjcf):
        pass

    def post_compilation(self):
        # Arm joint and actuator IDs.
        arm_joint_names = [j.full_identifier for j in self._arm_jnts]
        self._arm_joint_ids = np.asarray([self._model.joint(name).id for name in arm_joint_names])
        actuator_names = [a.full_identifier for a in self._arm_acts]
        self._arm_actuator_ids = np.asarray([self._model.actuator(name).id for name in actuator_names])
        gripper_actuator_names = [a.full_identifier for a in self._gripper_acts]
        self._gripper_actuator_ids = np.asarray([self._model.actuator(name).id for name in gripper_actuator_names])
        self._gripper_opening_joint_id = self._model.joint('ur5e/robotiq/right_driver_joint').id

        # Modify PD gains.
        self._model.actuator_gainprm[self._arm_actuator_ids, 0] = np.asarray([4500, 4500, 4500, 2000, 2000, 500])
        self._model.actuator_gainprm[self._arm_actuator_ids, 2] = np.asarray([-450, -450, -450, -200, -200, -50])
        self._model.actuator_biasprm[self._arm_actuator_ids, 1] = -np.asarray([4500, 4500, 4500, 2000, 2000, 500])

        # Site IDs.
        self._pinch_site_id = self._model.site('ur5e/robotiq/pinch').id
        self._attach_site_id = self._model.site('ur5e/attachment_site').id

        pinch_pose = lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_matrix(self._data.site_xmat[self._pinch_site_id].reshape(3, 3)),
            translation=self._data.site_xpos[self._pinch_site_id],
        )
        attach_pose = lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_matrix(self._data.site_xmat[self._attach_site_id].reshape(3, 3)),
            translation=self._data.site_xpos[self._attach_site_id],
        )
        self._T_pa = pinch_pose.inverse() @ attach_pose

        self.post_compilation_objects()

    def post_compilation_objects(self):
        pass

    def reset(self, options=None, *args, **kwargs):
        # `super().reset` runs `initialize_episode`, which calls `set_new_target` for the first segment.
        self._segment_index = -1
        self._segment_step = 0

        if self._mode == 'task':
            # Set the task goal.
            if options is None:
                options = {}

            if self._reward_task_id is not None:
                # Use the pre-defined task.
                assert 1 <= self._reward_task_id <= self.num_tasks, f'Task ID must be in [1, {self.num_tasks}].'
                self.cur_task_id = self._reward_task_id
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]
            elif 'task_id' in options:
                # Use the pre-defined task.
                assert 1 <= options['task_id'] <= self.num_tasks, f'Task ID must be in [1, {self.num_tasks}].'
                self.cur_task_id = options['task_id']
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]
            elif 'task_info' in options:
                # Use the provided task information.
                self.cur_task_id = None
                self.cur_task_info = options['task_info']
            else:
                # Randomly sample a task.
                self.cur_task_id = np.random.randint(1, self.num_tasks + 1)
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]

            # Whether to provide a rendering of the goal.
            self._render_goal = False
            if 'render_goal' in options:
                self._render_goal = options['render_goal']

        return super().reset(*args, **kwargs)

    def step(self, action):
        if self._reset_next_step:
            return self.reset()

        if self._success_timing == 'pre':
            success = self._success
            terminated = self.terminate_episode()
            reward = self.compute_reward()

        action = np.array(action)
        self.set_control(action)
        self.pre_step()
        mujoco.mj_step(self._model, self._data, nstep=self._n_steps)
        mujoco.mj_rnePostConstraint(self._model, self._data)  # Compute contact forces.
        self.post_step()

        if self._success_timing == 'post':
            success = self._success
            terminated = self.terminate_episode()
            reward = self.compute_reward()

        truncated = self.truncate_episode()
        ob = self.compute_observation()
        info = self.get_step_info()

        info['success'] = success

        return ob, reward, terminated, truncated, info

    def initialize_arm(self):
        # Sample initial effector position and orientation.
        eff_pos = self.np_random.uniform(*self._arm_sampling_bounds)
        cur_ori = self._effector_down_rotation
        yaw = self.np_random.uniform(-np.pi, np.pi)
        rotz = lie.SO3.from_z_radians(yaw)
        eff_ori = rotz @ cur_ori

        # Solve for initial joint positions using IK.
        T_wp = lie.SE3.from_rotation_and_translation(eff_ori, eff_pos)
        T_wa = T_wp @ self._T_pa
        qpos_init = self._ik.solve(
            pos=T_wa.translation(),
            quat=T_wa.rotation().wxyz,
            curr_qpos=self._home_qpos,
        )

        self._data.qpos[self._arm_joint_ids] = qpos_init
        mujoco.mj_forward(self._model, self._data)

    def initialize_episode(self):
        pass

    def set_new_target(self, return_info=True):
        pass

    def begin_new_segment(self):
        """Mark the start of a new subtask segment.

        Called by every `set_new_target` implementation. `_segment_step` is incremented in `post_step`, so the first
        step of a segment reports `_segment_step == 1`; that is what `segment_start` keys off. Deriving the flag from
        a counter rather than mutating state at read time keeps `compute_ob_info` free of side effects, which matters
        because `pre_step` also calls it.
        """
        self._segment_index += 1
        self._segment_step = 0

    def post_step(self):
        self._segment_step += 1

    def set_control(self, action):
        action = self.unnormalize_action(action)
        a_pos, a_ori, a_gripper = action[:3], action[3], action[4]

        # Compute target effector pose based on the relative action.
        effector_pos = self._data.site_xpos[self._pinch_site_id].copy()
        effector_yaw = lie.SO3.from_matrix(
            self._data.site_xmat[self._pinch_site_id].copy().reshape(3, 3)
        ).compute_yaw_radians()
        gripper_opening = np.array(np.clip([self._data.qpos[self._gripper_opening_joint_id] / 0.8], 0, 1))
        target_effector_translation = effector_pos + a_pos
        target_effector_orientation = (
            lie.SO3.from_z_radians(a_ori)
            @ lie.SO3.from_z_radians(effector_yaw)
            @ self._effector_down_rotation.inverse()
        )
        target_gripper_opening = gripper_opening + a_gripper

        # Make sure the target pose respects the action limits.
        np.clip(
            target_effector_translation,
            *self._workspace_bounds,
            out=target_effector_translation,
        )
        yaw = np.clip(
            target_effector_orientation.compute_yaw_radians(),
            -np.pi,
            +np.pi,
        )
        target_effector_orientation = lie.SO3.from_z_radians(yaw) @ self._effector_down_rotation
        target_gripper_opening = np.clip(target_gripper_opening, 0.0, 1.0)

        # Pinch pose in the world frame -> attach pose in the world frame.
        self._target_effector_pose = lie.SE3.from_rotation_and_translation(
            rotation=target_effector_orientation,
            translation=target_effector_translation,
        )
        T_wa = self._target_effector_pose @ self._T_pa

        # Solve for the desired joint positions.
        qpos_target = self._ik.solve(
            pos=T_wa.translation(),
            quat=T_wa.rotation().wxyz,
            curr_qpos=self._data.qpos[self._arm_joint_ids],
        )

        # Set the desired joint positions for the underlying PD controller.
        self._data.ctrl[self._arm_actuator_ids] = qpos_target
        self._data.ctrl[self._gripper_actuator_ids] = 255.0 * target_gripper_opening

    def pre_step(self):
        self._prev_qpos = self._data.qpos.copy()
        self._prev_qvel = self._data.qvel.copy()
        self._prev_ob_info = self.compute_ob_info()

    def compute_ob_info(self):
        ob_info = {}

        # Proprioceptive observations
        ob_info['proprio/joint_pos'] = self._data.qpos[self._arm_joint_ids].copy()
        ob_info['proprio/joint_vel'] = self._data.qvel[self._arm_joint_ids].copy()
        ob_info['proprio/effector_pos'] = self._data.site_xpos[self._pinch_site_id].copy()
        ob_info['proprio/effector_yaw'] = np.array(
            [lie.SO3.from_matrix(self._data.site_xmat[self._pinch_site_id].copy().reshape(3, 3)).compute_yaw_radians()]
        )
        ob_info['proprio/gripper_opening'] = np.array(
            np.clip([self._data.qpos[self._gripper_opening_joint_id] / 0.8], 0, 1)
        )
        ob_info['proprio/gripper_vel'] = self._data.qvel[[self._gripper_opening_joint_id]].copy()
        ob_info['proprio/gripper_contact'] = np.array(
            [np.clip(np.linalg.norm(self._data.body('ur5e/robotiq/right_pad').cfrc_ext) / 50, 0, 1)]
        )

        self.add_object_info(ob_info)

        ob_info['prev_qpos'] = self._prev_qpos.copy()
        ob_info['prev_qvel'] = self._prev_qvel.copy()
        ob_info['qpos'] = self._data.qpos.copy()
        ob_info['qvel'] = self._data.qvel.copy()
        ob_info['control'] = self._data.ctrl.copy()
        ob_info['time'] = np.array([self._data.time])

        return ob_info

    def add_object_info(self, ob_info):
        pass

    def get_pixel_observation(self):
        frame = self.render()
        return frame

    def compute_observation(self):
        if self._ob_type == 'pixels':
            return self.get_pixel_observation()
        else:
            xyz_center = np.array([0.425, 0.0, 0.0])
            xyz_scaler = 10.0
            gripper_scaler = 3.0

            ob_info = self.compute_ob_info()
            ob = [
                ob_info['proprio/joint_pos'],
                ob_info['proprio/joint_vel'],
                (ob_info['proprio/effector_pos'] - xyz_center) * xyz_scaler,
                np.cos(ob_info['proprio/effector_yaw']),
                np.sin(ob_info['proprio/effector_yaw']),
                ob_info['proprio/gripper_opening'] * gripper_scaler,
                ob_info['proprio/gripper_contact'],
            ]

            return np.concatenate(ob)

    def compute_reward(self):
        return 1.0 if self._success else 0.0

    def get_reset_info(self):
        reset_info = self.compute_ob_info()
        if self._mode == 'task':
            reset_info['goal'] = self._cur_goal_ob
            if self._render_goal is not None:
                reset_info['goal_rendered'] = self._cur_goal_rendered
        return reset_info

    def get_step_info(self):
        ob_info = self.compute_ob_info()
        return ob_info

    def terminate_episode(self):
        if self._terminate_at_goal:
            return self._success
        else:
            return False

    def render(
        self,
        camera=None,
        *args,
        **kwargs,
    ):
        if camera is None:
            camera = 'front' if self._ob_type == 'states' else 'front_pixels'

        return super().render(camera=camera, *args, **kwargs)

    @property
    def render_lighting(self):
        """The lighting configuration in effect, or None for upstream lighting. Record this in the dataset."""
        return None if self._render_lighting is None else dict(self._render_lighting)

    @staticmethod
    def resolve_camera_name(name):
        """Map a friendly camera name onto its compiled MJCF identifier, passing through unknown names."""
        return CAMERA_ALIASES.get(name, name)

    def render_cameras(self, camera_names=None, *args, **kwargs):
        """Render one frame per camera at the environment's render resolution.

        Args:
            camera_names: Cameras to render, as friendly names (see `CAMERA_ALIASES`). Defaults to
                `render_camera_names` given at construction.

        Returns:
            A dict keyed by the friendly camera name, with (height, width, 3) uint8 frames.
        """
        if camera_names is None:
            camera_names = self._render_camera_names
        if camera_names is None:
            raise ValueError('No cameras to render; pass render_camera_names to the environment or camera_names here.')

        return {name: self.render(camera=self.resolve_camera_name(name), *args, **kwargs) for name in camera_names}
