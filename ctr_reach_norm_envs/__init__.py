from gym.envs.registration import register
import numpy as np

register(
    id='CTR-Reach-Norm-v0', entry_point='ctr_reach_norm_envs.envs:CtrReachNormEnv',
    kwargs={
        'ctr_systems_parameters': {
            # CRL Dataset parameters
            'ctr_0': {
                'tube_0':
                    {'length': 110.0e-3, 'length_curved': 100e-3, 'diameter_inner': 1.2e-3, 'diameter_outer': 1.5e-3,
                     'stiffness': 50e+10, 'torsional_stiffness': 50.0e+10 / (2 * (1 + 0.3)),
                     'x_curvature': 4.37, 'y_curvature': 0},
                'tube_1':
                    {'length': 165.0e-3, 'length_curved': 100e-3, 'diameter_inner': 0.7e-3, 'diameter_outer': 0.9e-3,
                     'stiffness': 50e+10, 'torsional_stiffness': 50.0e+10 / (2 * (1 + 0.3)),
                     'x_curvature': 12.4, 'y_curvature': 0},
                'tube_2':
                    {'length': 210.0e-3, 'length_curved': 31e-3, 'diameter_inner': 0.4e-3, 'diameter_outer': 0.5e-3,
                     'stiffness': 50e+10, 'torsional_stiffness': 50.0e+10 / (2 * (1 + 0.3)),
                     'x_curvature': 28.0, 'y_curvature': 0},
            }
        },
        'extension_action_limit': 0.001,
        'rotation_action_limit': 5,
        'max_steps_per_episode': 150,
        'n_substeps': 10,
        'goal_tolerance_parameters': {
            'inc_tol_obs': False, 'final_tol': 0.001, 'initial_tol': 0.020,
            'N_ts': 200000, 'function': 'constant', 'set_tol': 0
        },
        'noise_parameters': {
            # 0.001 is the gear ratio
            # 0.001 is also the tracking std deviation for now for testing.
            'rotation_std': np.deg2rad(0), 'extension_std': 0.001 * np.deg2rad(0), 'tracking_std': 0.0
        },
        'select_systems': [0],
        # Format is [beta_0, beta_1, ..., beta_n, alpha_0, ..., alpha_n]
        #'initial_joints': np.array([-110e-3, -165e-3, -210e-3, 0, 0, 0]),
        'initial_joints': np.array([0., 0., 0., 0., 0., 0.]),
        'resample_joints': False,
        'evaluation': False,
        'length_based_sample': False,
        'domain_rand': 0.0
    },
)
