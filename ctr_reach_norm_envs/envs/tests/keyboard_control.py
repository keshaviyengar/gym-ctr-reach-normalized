from pynput.keyboard import Key, Listener
import gym
import time
import numpy as np

import ctr_reach_norm_envs

from matplotlib import pyplot as plt

class KeyboardControl(object):
    def __init__(self, env):
        self.env = env

        self.key_listener = Listener(on_press=self.on_press_callback)
        self.key_listener.start()

        self.action = np.zeros_like(self.env.action_space.low)
        self.extension_actions = np.zeros(3)
        self.rotation_actions = np.zeros(3)

        self.extension_value = self.env.action_space.high[0] / 2
        self.rotation_value = self.env.action_space.high[-1] / 2
        self.exit = False

    def on_press_callback(self, key):
        # Tube 1 (outermost tube) is w s a d
        # Tube 2 (middle tube) is t g f h
        # Tube 3 (innermost tube) is i k j l
        try:
            if key.char in ['1', '2', '3', '4']:
                if key.char == '1':
                    self.extension_actions[0] = self.extension_value
                elif key.char == '2':
                    self.extension_actions[0] = -self.extension_value
                elif key.char == '3':
                    self.rotation_actions[0] = self.rotation_value
                elif key.char == '4':
                    self.rotation_actions[0] = -self.rotation_value
            if key.char in ['5', '6', '7', '8']:
                if key.char == '5':
                    self.extension_actions[1] = self.extension_value
                elif key.char == '6':
                    self.extension_actions[1] = -self.extension_value
                elif key.char == '7':
                    self.rotation_actions[1] = self.rotation_value
                elif key.char == '8':
                    self.rotation_actions[1] = -self.rotation_value
            if key.char in ['z', 'x', 'c', 'v']:
                if key.char == 'z':
                    self.extension_actions[2] = self.extension_value
                elif key.char == 'x':
                    self.extension_actions[2] = -self.extension_value
                elif key.char == 'c':
                    self.rotation_actions[2] = self.rotation_value
                elif key.char == 'v':
                    self.rotation_actions[2] = -self.rotation_value
        except AttributeError:
            if key == Key.esc:
                self.exit = True
                exit()
            else:
                self.extension_actions = np.zeros(3)
                self.rotation_actions = np.zeros(3)

    def run(self):
        obs = self.env.reset()
        while not self.exit:
            self.action[:3] = self.extension_actions
            self.action[3:] = self.rotation_actions
            observation, reward, done, info = self.env.step(self.action)
            self.extension_actions = np.zeros(3)
            self.rotation_actions = np.zeros(3)
            self.action = np.zeros_like(self.env.action_space.low)
            self.env.render('live')
        self.env.close()


if __name__ == '__main__':
    spec = gym.spec('CTR-Reach-Norm-v0')
    kwargs = {}
    env = spec.make(**kwargs)
    keyboard_agent = KeyboardControl(env)
    keyboard_agent.run()