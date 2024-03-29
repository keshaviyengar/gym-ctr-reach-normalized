import gym
import numpy as np
from ctr_reach_norm_envs.envs.obs import Obs
from ctr_reach_norm_envs.envs.goal_tolerance import GoalTolerance
from ctr_reach_norm_envs.envs.ctr_3d_graph import Ctr3dGraph
from ctr_reach_norm_envs.envs.model import Model
from ctr_reach_norm_envs.envs.CTR_Python import Tube

NUM_TUBES = 3


def out2in_convention(in2out_joints):
    return np.concatenate((np.flip(in2out_joints[:3]), np.flip(in2out_joints[3:])))


def in2out_convention(out2in_joints):
    return np.concatenate((np.flip(out2in_joints[:3]), np.flip(out2in_joints[3:])))


class CtrReachNormEnv(gym.GoalEnv):
    def __init__(self, ctr_systems_parameters, goal_tolerance_parameters, noise_parameters,
                 initial_joints, extension_action_limit, rotation_action_limit, max_steps_per_episode,
                 n_substeps, evaluation, select_systems, resample_joints=True, length_based_sample=False,
                 domain_rand=0.0):

        # Load in all system parameters
        self.ctr_system_parameters = list()
        for system in ctr_systems_parameters:
            tubes = list()
            for tube in ctr_systems_parameters[system]:
                tubes.append(Tube(**ctr_systems_parameters[system][tube]))
            self.ctr_system_parameters.append(tubes)

        # Remove unused systems based of select_systems
        self.select_systems = select_systems
        self.ctr_system_parameters = [self.ctr_system_parameters[system] for system in select_systems]
        self.noise_parameters = noise_parameters
        self.max_steps_per_episode = max_steps_per_episode
        self.n_substeps = n_substeps
        # Other parameters and settings
        self.starting_joints = initial_joints
        self.desired_joints = initial_joints
        self.evaluation = evaluation
        self.resample_joints = resample_joints
        self.length_based_sample = length_based_sample
        self.domain_rand = domain_rand

        # CTR kinematic model
        # Reverse order of tube for model (innermost to outermost)
        ctr_systems_model = self.ctr_system_parameters.copy()
        ctr_systems_model[0].reverse()
        self.model = Model(ctr_systems_model)

        self.visualization = None

        # Initialization parameters / objects
        self.t = 0
        ctr_systems_model[0].reverse()
        self.trig_obj = Obs(ctr_systems_model, goal_tolerance_parameters, noise_parameters, initial_joints)
        self.observation_space = self.trig_obj.get_observation_space()

        self.extension_action_limit = extension_action_limit
        self.rotation_action_limit = rotation_action_limit
        self.action_space = gym.spaces.Box(low=-np.ones(2 * NUM_TUBES), high=np.ones(2 * NUM_TUBES))
        self.system = 0
        # Initialization of starting position, need to convert from outer first representation to inner first
        self.starting_position = self.model.forward_kinematics(in2out_convention(np.array(self.starting_joints)), self.system)
        self.desired_goal = self.starting_position
        # Goal tolerance parameters
        self.goal_tolerance = GoalTolerance(goal_tolerance_parameters)

    def reset(self, goal=None, system=None):
        """
        Reset function as specified by gym. Called at the start of each episode.
        :param goal: Give the agent a specific goal to reach. Set to None if sample a new desired goal.
        :param system: The CTR system to use for this episode.
        :return: The observation or achieved, desired goals and joints in trigonometric representation
        """
        # Reset timesteps
        self.t = 0
        # Domain randomization for domain transfer. Set self.domain_rand to zero if not used
        self.model.randomize_parameters(self.domain_rand)
        # Set system to None if training and want to sample systems at each episode
        if system is None:
            # Use non-uniform sampling based on length of each system
            if self.length_based_sample:
                # Get overall lengths of systems
                all_system_length = 0
                system_length = []
                for system in self.ctr_system_parameters:
                    all_system_length += system[0].L
                    system_length.append(system[0].L)
                sys_prob = np.array(system_length) / all_system_length
                self.system = np.where(np.random.multinomial(1, sys_prob) == 1)[0][0]
            else:
                # Sample uniformly
                self.system = np.random.randint(len(self.ctr_system_parameters))
        else:
            self.system = system
        if goal is None:
            # No goal given so sample a desired goal in the robot workspace
            self.desired_joints = self.trig_obj.sample_goal(self.system)
            self.desired_goal = self.model.forward_kinematics(in2out_convention(self.desired_joints), self.system)
        else:
            self.desired_goal = goal
        if self.resample_joints:
            # Should the initial position of the robot be re-sampled at the start of each episode
            self.starting_joints = self.trig_obj.sample_goal(self.system)
            self.trig_obj.joints = self.starting_joints
            self.starting_position = self.model.forward_kinematics(in2out_convention(self.trig_obj.joints), self.system)
        else:
            # Start from the final position of last episode
            self.starting_joints = self.trig_obj.joints
            self.starting_position = self.model.forward_kinematics(in2out_convention(self.starting_joints), self.system)
        obs = self.trig_obj.get_obs(self.desired_goal, self.starting_position, self.goal_tolerance.get_tol(),
                                    self.system)
        return obs

    def seed(self, seed=None):
        """
        Set the seed
        :param seed: Value of seed. If None, sample one.
        """
        if seed is not None:
            np.random.seed(seed)

    def step(self, action):
        """
        Step function as defined by gym. Takes input of action and simulates the environment by one step.
        :param action: Selected action or changes in joint values.
        :return: New observation of desired, achieved goals and trigonometric joint representation
        """
        # Ensure actions are not NaNs and within action space to enforce constraints
        assert not np.all(np.isnan(action))
        assert self.action_space.contains(action)
        action[:3] = action[:3] * self.extension_action_limit
        action[3:] = action[3:] * np.deg2rad(self.rotation_action_limit)
        # For n_substeps, repeat the selected action
        for i in range(self.n_substeps):
            self.trig_obj.set_action(action, self.system)
        # Compute achieved goal with forward kinematics
        achieved_goal = self.model.forward_kinematics(in2out_convention(np.array(self.trig_obj.joints)), self.system)
        self.t += 1
        reward = self.compute_reward(achieved_goal, self.desired_goal, dict())
        done = (reward == 0) or (self.t >= self.max_steps_per_episode)
        obs = self.trig_obj.get_obs(self.desired_goal, achieved_goal, self.goal_tolerance.get_tol(), self.system)

        # If evaluating, save more information for analysis
        if self.evaluation:
            info = {'is_success': (np.linalg.norm(self.desired_goal - achieved_goal) < self.goal_tolerance.get_tol()),
                    'errors_pos': np.linalg.norm(self.desired_goal - achieved_goal),
                    'errors_orient': 0,
                    'system_idx': self.select_systems[self.system],
                    'position_tolerance': self.goal_tolerance.get_tol(),
                    'orientation_tolerance': 0,
                    'achieved_goal': achieved_goal,
                    'desired_goal': self.desired_goal, 'starting_position': self.starting_position,
                    'q_desired': self.desired_joints, 'q_achieved': self.trig_obj.joints,
                    'q_starting': self.starting_joints}
        else:
            info = {'is_success': (np.linalg.norm(self.desired_goal - achieved_goal) < self.goal_tolerance.get_tol()),
                    'error': np.linalg.norm(self.desired_goal - achieved_goal)}
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Compute the reward given current and desired goals based on Euclidean distance.
        :param achieved_goal: Current achieved position of end-effector.
        :param desired_goal: Desired position of end-effector.
        :param info: Dictionary for extra details.
        :return: -1 or 0 based on current tolerance.
        """
        assert achieved_goal.shape == desired_goal.shape
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if d < self.goal_tolerance.get_tol():
            return 0.0
        else:
            return -d

    def render(self, mode='empty', **kwargs):
        """
        Render the current shape of the CTR system with achieved goal and desired goal.
        :param mode: Set the render mode. If not set, no rendering is performed.
        :param kwargs: Extra arguements if needed.
        """
        if mode == 'live':
            if self.visualization is None:
                self.visualization = Ctr3dGraph()
            self.visualization.render(self.t, self.trig_obj.obs['achieved_goal'], self.trig_obj.obs['desired_goal'],
                                      self.model.r1, self.model.r2, self.model.r3)

    def close(self):
        """
        Close gym environment.
        """
        if self.visualization != None:
            self.visualization.close()
            self.visualization = None

    def update_goal_tolerance(self, timestep):
        """

        :param timestep:  The current timestep to update the goal tolerance
        """
        self.goal_tolerance.update(timestep)

    def get_goal_tolerance(self):
        return self.goal_tolerance.get_tol()
