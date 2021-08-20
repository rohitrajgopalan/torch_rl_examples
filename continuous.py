import gym
import numpy as np

try:
    from common import run_actor_critic_continuous_methods
except ImportError:
    from .common import run_actor_critic_continuous_methods

env_continuous = gym.make('LunarLanderContinuous-v2')
run_actor_critic_continuous_methods(env_continuous, 'lunar_lander')

env_continuous = gym.make('MountainCarContinuous-v0')
run_actor_critic_continuous_methods(env_continuous, 'mountain_car', np.array([0.45, 0]))

env = gym.make('BipedalWalker-v3')
run_actor_critic_continuous_methods(env, 'bipedal_walker')

env = gym.make('gym_ccc.envs:Multirotor2DSimpNonNormCont-v0')
run_actor_critic_continuous_methods(env, 'multirotor', np.array([5, 5, 0, 0, 0]))
