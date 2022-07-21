import gym
import ctr_reach_norm_envs
from stable_baselines.common.env_checker import check_env
import numpy as np

if __name__ == '__main__':
    # Check environment
    sb_check_env = False
    env = gym.make(id='CTR-Reach-Norm-v0')
    if sb_check_env:
        check_env(env=env)

    # Run an episode with rendering
    obs = env.reset()
    # Rotate inner tube
    for _ in range(int(150 / 5)):
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        obs, reward, done, info =  env.step(action)
        env.render('live')
        assert env.observation_space.contains(obs)

    # Extend inner, middle and outer tube
