import gym
try:
    from common import run_actor_critic_continuous_methods
except ImportError:
    from .common import run_actor_critic_continuous_methods

env = gym.make('BipedalWalker-v3')
run_actor_critic_continuous_methods(env, 'bipedal_walker')

env_hardcore  = gym.make('BipdealWalkerHardcore-v3')
run_actor_critic_continuous_methods(env_hardcore, 'bipedal_walker_hardcore')

