import gym
try:
    from common import run_all_discrete_methods, run_actor_critic_continuous_methods
except ImportError:
    from .common import run_all_discrete_methods, run_actor_critic_continuous_methods

env_discrete = gym.make('MountainCar-v0')
run_all_discrete_methods(env_discrete, 'mountain_car')

env_continuous = gym.make('MountainCarContinuous-v0')
run_actor_critic_continuous_methods(env_continuous, 'mountain_car')