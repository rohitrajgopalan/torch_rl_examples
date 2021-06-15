import gym
try:
    from common import run_all_discrete_methods, run_actor_critic_continuous_methods
except ImportError:
    from .common import run_all_discrete_methods, run_actor_critic_continuous_methods

env = gym.make('Acrobot-v1')
run_all_discrete_methods(env, 'acrobot')

env = gym.make('CartPole-v1')
run_all_discrete_methods(env, 'cartpole')

env_discrete = gym.make('LunarLander-v2')
run_all_discrete_methods(env_discrete, 'lunar_lander')

env_discrete = gym.make('MountainCar-v0')
run_all_discrete_methods(env_discrete, 'mountain_car')

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