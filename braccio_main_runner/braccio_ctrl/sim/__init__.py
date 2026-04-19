"""
braccio_ctrl.sim — PyBullet simulation environment for RL training.

Modules
-------
braccio_env   : BraccioSimEnv — Gymnasium env backed by PyBullet physics.
                Inherits BraccioBaseEnv from rl_env.py.
noise_params  : load_noise_params() — load calibrated noise JSON.

Usage
-----
from braccio_ctrl.sim.braccio_env import BraccioSimEnv
env = BraccioSimEnv()
obs, _ = env.reset()
obs, reward, term, trunc, info = env.step(env.action_space.sample())
"""
