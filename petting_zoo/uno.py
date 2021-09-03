from pettingzoo.classic import uno_v4
from common.common_pettingzoo_observation import run_all_td_methods

env = uno_v4.env()
run_all_td_methods(env, 'uno', 1)

