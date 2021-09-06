import gym

from common.common_gym_observation import run_all_td_methods, run_heuristics, run_hill_climbing

env = gym.make('Blackjack-v0')


def blackjack_heuristic(self, observation):
    total, dealer_card, _ = observation
    if total >= 17 or (13 <= total <= 16 and 2 <= dealer_card <= 6) or (total == 12 and 4 <= dealer_card <= 6):
        return 0
    else:
        return 1


run_all_td_methods(env, 'blackjack', 0)
run_hill_climbing(env, 'blackjack', 0)
run_heuristics(env, 'blackjack', penalty=0, heuristic_func=blackjack_heuristic)
