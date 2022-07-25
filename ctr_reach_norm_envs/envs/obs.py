import numpy as np
import gym

from ctr_reach_norm_envs.envs.goal_tolerance import GoalTolerance
from ctr_reach_norm_envs.envs.obs_utils import *

'''
The Obs class defines the observation space and defines functions to create a obs given joint values, desired and
achieved goals. Also sets actions with constraints to the system.
'''

NUM_TUBES = 3
# Extension tolerance TODO: Maybe not needed
EXT_TOL = 1e-3
# Prevent reaching maximum retraction position with causes issues in model
ZERO_TOL = 1e-4


class Obs(object):
    def __init__(self, system_parameters, goal_tolerance_parameters, noise_parameters, initial_joints):
        self.system_parameters = system_parameters
        self.num_systems = len(self.system_parameters)
        # Get tube lengths to set maximum beta value in observation space
        self.tube_lengths = list()
        for sys_param in self.system_parameters:
            tube_length = list()
            for tube in sys_param:
                tube_length.append(tube.L)
            self.tube_lengths.append(tube_length)

        # Goal tolerance parameters
        self.goal_tolerance = GoalTolerance(goal_tolerance_parameters)

        # Noise parameters used for noise induced simulations
        extension_std_noise = np.full(NUM_TUBES, noise_parameters['extension_std'])
        rotation_std_noise = np.full(NUM_TUBES, noise_parameters['rotation_std'])
        self.q_std_noise = np.concatenate((extension_std_noise, rotation_std_noise))
        self.tracking_std_noise = np.full(3, noise_parameters['tracking_std'])

        # Variables for joint values and relative joint values
        self.joints = np.array(initial_joints, dtype='float16')
        self.joint_spaces, joint_sample_space = self.get_joint_space()
        self.observation_space = self.get_observation_space()

    def get_joint_space(self):
        """
        Get the joint space to constrain rotation and extension to limits. Separately, create joint_sample_space from
        which samples are taken to get desired goals in the robot workspace.
        :return: The joint space and sample joint space.
        """
        joint_spaces = list()
        joint_sample_spaces = list()
        for tube_betas in self.tube_lengths:
            joint_spaces.append(gym.spaces.Box(low=np.concatenate((-np.array(tube_betas) + EXT_TOL,
                                                                   np.full(NUM_TUBES, -np.inf))),
                                               high=np.concatenate((np.full(NUM_TUBES, 0),
                                                                    np.full(NUM_TUBES, np.inf)))
                                               , dtype="float32"))
        return joint_spaces, joint_sample_spaces

    def get_observation_space(self):
        """
        Get the observation space defining the limits of the obs as defined in gym
        :return: The observation space
        """
        rep_space = gym.spaces.Box(low=-1.0 * np.ones(9), high=1.0 * np.ones(9), dtype="float32")

        # If training a single system, don, include the psi variable indicating system
        if self.num_systems == 1:
            # NUM_TUBES * (sin(alpha), cos(alpha), beta),  del_x, del_y, del_z, initial_tol
            obs_space_low = np.concatenate((rep_space.low, -np.ones(4)))
            obs_space_high = np.concatenate((rep_space.high, np.ones(4)))
        else:
            # del_x, del_y, del_z, initial_tol, psi_system_variable
            obs_space_low = np.concatenate((rep_space.low, -np.ones(5)))
            obs_space_high = np.concatenate((rep_space.high, np.ones(5)))

        observation_space = gym.spaces.Dict(dict(
            desired_goal=gym.spaces.Box(low=-np.ones(3), high=np.ones(3), dtype="float32"),
            achieved_goal=gym.spaces.Box(low=-np.ones(3), high=np.ones(3), dtype="float32"),
            observation=gym.spaces.Box(low=obs_space_low, high=obs_space_high, dtype="float32")
        ))
        return observation_space

    def get_obs(self, desired_goal, achieved_goal, goal_tolerance, system):
        """
        The an observation object given the current desired, achieved goals, goal tolerance and system selected
        :param desired_goal: Current desired goal
        :param achieved_goal: Current achieved goal of the end-effector
        :param goal_tolerance: Current goal tolerance
        :param system: Selected system
        :return: Observation object
        """
        # Joints using new referencing (outermost to innermost)
        betas_U = B_to_B_U(self.joints[:NUM_TUBES], self.tube_lengths[system][0], self.tube_lengths[system][1],
                           self.tube_lengths[system][2])
        if np.any(betas_U) < -1.0 or np.any(betas_U) > 1.0:
            print("betas_U not correct.")

        trig_joints = joint2rep(np.concatenate((betas_U, self.joints[NUM_TUBES:])))
        normalized_error = normalize(np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5]),
                                     desired_goal - achieved_goal)
        normalized_goal_tolerance = normalize(self.goal_tolerance.final_tol, self.goal_tolerance.init_tol,
                                              goal_tolerance)
        normalized_system = normalize(0, self.num_systems, system)
        if self.num_systems > 1:
            obs = np.concatenate(
                [trig_joints, normalized_error, np.array([normalized_goal_tolerance, normalized_system])])
        else:
            obs = np.concatenate(
                [trig_joints, normalized_error, np.array([normalized_goal_tolerance])])
        if not self.observation_space['observation'].contains(obs):
            pass
            #print('obs not in space. ' + str(obs[obs > 1.0] or obs < -1.0))
        self.obs = {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': desired_goal.copy()
        }
        return self.obs

    def set_action(self, action, system):
        """
                irint(betas_U)
        Set the given action to the joints for the selected system.
        :param action: Array of rotation and extensions for each tube.
        :param system: The selected system.
        """
        if not np.any(action) == 0.:
            new_joints = np.clip(np.around(self.joints + action, 6), self.joint_spaces[system].low, self.joint_spaces[system].high)
            betas = new_joints[:NUM_TUBES]
            alphas = new_joints[NUM_TUBES:]
            betas_U = B_to_B_U(betas, self.tube_lengths[system][0], self.tube_lengths[system][1], self.tube_lengths[system][2])
            if not np.any(betas_U < -1.0) and not np.any(betas_U > 1.0):
                self.joints = np.hstack((betas, alphas))
        #for i in range(len(betas)-2, -2, -1):
        #    print('b0: ' + str(betas[0]))
        #    betas[0] = max(betas[0], self.tube_lengths[system][1] + betas[1] - self.tube_lengths[system][0])
        #    if i == -1:
        #        print('b0: ' + str(betas[0]))
        #        betas[0] = max(betas[0], self.tube_lengths[system][1] + betas[1] - self.tube_lengths[system][0])
        #        print('ab0: ' + str(betas[0]))
        #        betas[0] = min(betas[0], 0.0)
        #        print('bb0: ' + str(betas[0]))
        #        betas[0] = max(betas[0], -self.tube_lengths[system][0])
        #        print('cb0: ' + str(betas[0]))
        #    else:
        #        betas[i+1] = min(betas[i + 1], betas[i])
        #        betas[i+1] = max(betas[i + 1],
        #                           self.tube_lengths[system][i] - self.tube_lengths[system][i + 1] + betas[i])


    def sample_goal(self, system):
        """
        Sample a joint goal while considering constraints on extension and joint limits.
        :param system: The system to to sample the goal.
        :return: Constrained achievable joint values.
        """
        betas = B_U_to_B(np.random.uniform(low=-np.ones(3), high=np.ones(3)), self.system_parameters[system][0].L,
                         self.system_parameters[system][1].L, self.system_parameters[system][2].L)
        alphas = alpha_U_to_alpha(np.random.uniform(low=-np.ones(3), high=np.ones(3)), np.pi)
        return np.concatenate((betas, alphas))
