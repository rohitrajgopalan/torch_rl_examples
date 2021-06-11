import gym
try:
    from common import run_all_discrete_methods, run_actor_critic_continuous_methods
except ImportError:
    from .common import run_all_discrete_methods, run_actor_critic_continuous_methods

env_discrete = gym.make('LunarLander-v2')
run_all_discrete_methods(env_discrete, 'lunar_lander')

env_continuous = gym.make('LunarLanderContinuous-v2')
run_actor_critic_continuous_methods(env_continuous, 'lunar_lander')