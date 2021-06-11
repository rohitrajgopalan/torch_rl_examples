import gym
try:
    from common import run_actor_critic_continuous_methods
except ImportError:
    from .common import run_actor_critic_continuous_methods

env = gym.make('Pendulum-v0')
run_actor_critic_continuous_methods(env, 'pendulum')
