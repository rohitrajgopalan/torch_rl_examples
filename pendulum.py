import gym
try:
    from common import *
except ImportError:
    from .common import *

env = gym.make('Pendulum-v0')
