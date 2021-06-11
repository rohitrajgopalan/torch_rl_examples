import gym
try:
    from common import *
except ImportError:
    from .common import *

env = gym.make('CartPole-v1')
run_all_discrete_methods(env, 'cartpole')