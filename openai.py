import gym
import numpy as np
try:
    from common import run_all_discrete_methods, run_actor_critic_continuous_methods
except ImportError:
    from .common import run_all_discrete_methods, run_actor_critic_continuous_methods

env = gym.make('FrozenLake-v0')
run_all_discrete_methods(env, 'frozen_lake', 0, 15)

env = gym.make('FrozenLake8x8-v0')
run_all_discrete_methods(env, 'frozen_lake_8_by_8', 0, 63)

env = gym.make('Taxi-v3')
run_all_discrete_methods(env, 'taxi', 20)

env = gym.make('CliffWalking-v0')
run_all_discrete_methods(env, 'cliff_walking', 100, 11)

env = gym.make('Acrobot-v1')
run_all_discrete_methods(env, 'acrobot', 0)

env = gym.make('CartPole-v1')
run_all_discrete_methods(env, 'cartpole', 0)

env_discrete = gym.make('LunarLander-v2')
run_all_discrete_methods(env_discrete, 'lunar_lander', 100)

env_discrete = gym.make('MountainCar-v0')
run_all_discrete_methods(env_discrete, 'mountain_car', 0, np.array([0.5, 0.5]))

env_continuous = gym.make('LunarLanderContinuous-v2')
run_actor_critic_continuous_methods(env_continuous, 'lunar_lander')

env_continuous = gym.make('MountainCarContinuous-v0')
run_actor_critic_continuous_methods(env_continuous, 'mountain_car')

env = gym.make('Pendulum-v0')
run_actor_critic_continuous_methods(env, 'pendulum')

env = gym.make('BipedalWalker-v3')
run_actor_critic_continuous_methods(env, 'bipedal_walker')

env_hardcore = gym.make('BipdealWalkerHardcore-v3')
run_actor_critic_continuous_methods(env_hardcore, 'bipedal_walker_hardcore')