import gym
import numpy as np

try:
    from common import run_all_td_methods, run_actor_critic_continuous_methods
except ImportError:
    from .common import run_all_td_methods, run_actor_critic_continuous_methods

env = gym.make('FrozenLake-v0')
frozen_lake_4_x_4_matrix = {0: [1, 2], 1: [0, 2], 2: [0, 1, 2], 3: [0], 4: [1, 3], 6: [1, 3], 8: [1, 2], 9: [0, 1, 2],
                            10: [0, 1, 3], 13: [2, 3], 14: [0, 2, 3]
                            }
run_all_td_methods(env, 'frozen_lake', 0, 15, frozen_lake_4_x_4_matrix)

env = gym.make('FrozenLake8x8-v0')
frozen_lake_8_x_8_matrix = {0: [1, 2], 1: [0, 1, 2], 2:[0, 1, 2], 3: [0, 1, 2], 4: [0, 1, 2], 5: [0, 1, 2],
                            6: [0, 1, 2], 7: [0, 1], 8: [1, 2, 3], 9: [0, 1, 2, 3], 10: [0, 1, 2, 3], 11: [0, 2, 3],
                            12: [0, 1, 2, 3], 13: [0, 1, 2, 3], 14: [0, 1, 2, 3], 15: [0, 1, 3], 16: [1, 2, 3],
                            17: [0, 1, 2, 3], 18: [0, 1, 3], 20: [1, 2, 3], 21: [0, 2, 3], 22: [0, 1, 2, 3],
                            23: [0, 1, 3], 24: [1, 2, 3], 25: [0, 1, 2, 3], 26: [0, 1, 2, 3], 27: [0, 2],
                            28: [0, 1, 3], 30: [1, 2, 3], 31: [0, 1, 3], 32: [1, 2, 3], 33: [0, 2, 3], 34: [0, 3],
                            36: [1, 2, 3], 37: [0, 1, 2], 38: [0, 2, 3], 39: [0, 1, 3], 40: [1, 3],
                            43: [1, 2], 44: [0, 2, 3], 45: [0, 1, 3], 47: [0, 1, 3], 48: [1, 3],
                            50: [1, 2], 51: [0, 3], 53: [1, 3], 55: [1, 3], 56: [2, 3], 57: [0, 2], 58: [0, 3],
                            60: [2], 61: [0, 2, 3], 62: [2]}
run_all_td_methods(env, 'frozen_lake_8_by_8', 0, 63, frozen_lake_8_x_8_matrix)

env = gym.make('Taxi-v3')
run_all_td_methods(env, 'taxi', 20)

env = gym.make('CliffWalking-v0')
cliff_walking_matrix = {}
for s in range(48):
    if s == 0:
        actions = [0, 1]
    elif s == 36:
        actions = [3]
    elif s == 11:
        actions = [1, 2]
    elif s in range(1, 11):
        actions = [0, 1, 2]
    elif s in range(25, 35):
        actions = [0, 2, 3]
    elif s % 12 == 0:
        actions = [0, 1, 3]
    elif (s + 1) % 12 == 0:
        actions = [1, 2, 3]
    else:
        actions = [0, 1, 2, 3]
    cliff_walking_matrix.update({s: actions})
run_all_td_methods(env, 'cliff_walking', 100, 47, cliff_walking_matrix)

# env = gym.make('CartPole-v1')
# run_all_td_methods(env, 'cartpole', 0)

env = gym.make('Blackjack-v0')
run_all_td_methods(env, 'black_jack', 1)

env_discrete = gym.make('LunarLander-v2')
run_all_td_methods(env_discrete, 'lunar_lander', 100)

env_discrete = gym.make('MountainCar-v0')
run_all_td_methods(env_discrete, 'mountain_car', 0, np.array([0.5, 0]))

# env_continuous = gym.make('LunarLanderContinuous-v2')
# run_actor_critic_continuous_methods(env_continuous, 'lunar_lander')
#
# env_continuous = gym.make('MountainCarContinuous-v0')
# run_actor_critic_continuous_methods(env_continuous, 'mountain_car')
