import gym

from common.common_gym_observation import run_all_td_methods, run_heuristics, run_hill_climbing

env = gym.make('CartPole-v1')
run_all_td_methods(env, 'cartpole', 0)
run_hill_climbing(env, 'cartpole', 0)


def cartpole_heuristic(self, observation):
    theta, omega = observation[2], observation[3]
    if abs(theta) < 0.03:
        return 0 if omega < 0 else 1
    else:
        return 0 if theta < 0 else 1


run_heuristics(env, 'cartpole', cartpole_heuristic)
