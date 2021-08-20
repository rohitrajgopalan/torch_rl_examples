import gym
import numpy as np

try:
    from common import run_all_td_methods
except ImportError:
    from .common import run_all_td_methods

env = gym.make('CartPole-v1')
run_all_td_methods(env, 'cartpole', 0)

env = gym.make('Blackjack-v0')
run_all_td_methods(env, 'black_jack', 0)

env_discrete = gym.make('LunarLander-v2')
run_all_td_methods(env_discrete, 'lunar_lander', 100)

env_discrete = gym.make('MountainCar-v0')
run_all_td_methods(env_discrete, 'mountain_car', 0, np.array([0.5, 0]))