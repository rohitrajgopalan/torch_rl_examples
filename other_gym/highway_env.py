import gym

from common.common_gym_observation import run_all_td_methods, run_actor_critic_continuous_methods


highway_env = gym.make("highway-v0")
run_all_td_methods(highway_env, 'highway', 0.01)

merge_env = gym.make("merge-v0")
run_all_td_methods(merge_env, 'merge', 0.01)

roundabout_env = gym.make("roundabout-v0")
run_all_td_methods(roundabout_env, 'roundabout', 0.01)

parking_env = gym.make("parking-v0")
run_actor_critic_continuous_methods(parking_env, 'parking')

intersection_env = gym.make("intersection-v0")
run_all_td_methods(intersection_env, 'intersection', 0.01)
