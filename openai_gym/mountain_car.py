import gym
import numpy as np

from common.common_gym_observation import run_actor_critic_continuous_methods, run_all_td_methods, run_heuristics, \
    run_hill_climbing, run_cem

env_discrete = gym.make('MountainCar-v0')
run_all_td_methods(env_discrete, 'mountain_car', 0, np.array([0.5, 0]))
run_hill_climbing(env_discrete, 'mountain_car', 0)

env_continuous = gym.make('MountainCarContinuous-v0')
run_actor_critic_continuous_methods(env_continuous, 'mountain_car', np.array([0.45, 0]))
run_cem(env_continuous, 'mountain_car', np.array([0.45, 0]))


def mountain_car_discrete_heuristic(self, observation):
    position, velocity = observation
    if position < 0:
        return 2 if velocity >= 1e-4 else 0
    else:
        return 0 if velocity >= 1e-4 else 2


run_heuristics(env_discrete, 'mountain_car_discrete', mountain_car_discrete_heuristic, 0, np.array([0.5, 0]))


def mountain_car_continuous_heuristic(self, observation):
    position, velocity = observation
    action = np.array([1.0])
    if position < 0:
        return action if velocity >= 1e-4 else -action
    else:
        return -action if velocity >= 1e-4 else action


run_heuristics(env_continuous, 'mountain_car_continuous', mountain_car_continuous_heuristic, 0, np.array([0.45, 0]))
