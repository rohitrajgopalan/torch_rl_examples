import gym
try:
    from common import run_all_discrete_methods
except ImportError:
    from .common import run_all_discrete_methods

env = gym.make('CartPole-v1')
run_all_discrete_methods(env, 'cartpole')