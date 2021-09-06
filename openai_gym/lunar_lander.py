import gym
import numpy as np

from common.common_gym_observation import run_actor_critic_continuous_methods, run_all_td_methods, run_heuristics, \
    run_hill_climbing, run_cem

env_discrete = gym.make('LunarLander-v2')
run_all_td_methods(env_discrete, 'lunar_lander', 100)
run_hill_climbing(env_discrete, 'lunar_lander', 100)

env_continuous = gym.make('LunarLanderContinuous-v2')
run_actor_critic_continuous_methods(env_continuous, 'lunar_lander')
run_cem(env_continuous, 'lunar_lander')


def lunar_lander_discrete_heuristic(self, observation):
    angle_targ = observation[0] * 0.5 + observation[2] * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(
        observation[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - observation[4]) * 0.5 - (observation[5]) * 1.0
    hover_todo = (hover_targ - observation[1]) * 0.5 - (observation[3]) * 0.5

    if observation[6] or observation[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (
                -(observation[3]) * 0.5
        )  # override to reduce fall speed, that's all we need after contact

    if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
        return 2
    elif angle_todo < -0.05:
        return 3
    elif angle_todo > +0.05:
        return 1
    else:
        return 0


run_heuristics(env_discrete, 'lunar_lander_discrete', lunar_lander_discrete_heuristic, 100)


def lunar_lander_continuous_heuristic(self, observation):
    angle_targ = observation[0] * 0.5 + observation[2] * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(
        observation[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - observation[4]) * 0.5 - (observation[5]) * 1.0
    hover_todo = (hover_targ - observation[1]) * 0.5 - (observation[3]) * 0.5

    if observation[6] or observation[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (
                -(observation[3]) * 0.5
        )  # override to reduce fall speed, that's all we need after contact

    a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
    return np.clip(a, -1, +1)


run_heuristics(env_continuous, 'lunar_lander_continuous', lunar_lander_continuous_heuristic)
