from pettingzoo.classic import texas_holdem_v3
from common.common_pettingzoo_observation import run_all_td_methods

env = texas_holdem_v3.env()
run_all_td_methods(env, 'texas_holdem', 1)
