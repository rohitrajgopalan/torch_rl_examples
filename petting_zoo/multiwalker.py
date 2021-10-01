import os

import pandas as pd
from torch_rl.cem.agent import CEMAgent
from torch_rl.heuristic.heuristic_with_cem import HeuristicWithCEM
from torch_rl.heuristic.heuristic_with_dt import HeuristicWithDT
from torch_rl.heuristic.heuristic_with_rf import HeuristicWithRF
from torch_rl.heuristic.heuristic_with_td3 import HeuristicWithTD3
from torch_rl.heuristic.heuristic_with_ddpg import HeuristicWithDDPG
from torch_rl.ddpg.agent import DDPGAgent
from torch_rl.td3.agent import TD3Agent
from torch_rl.utils.types import NetworkOptimizer, LearningType

from pettingzoo.sisl import multiwalker_v7

from common.run import run_pettingzoo_env
from common.utils import derive_hidden_layer_size, generate_agents_for_petting_zoo, develop_memory_from_pettingzoo_env
from openai_gym.bipedal_walker import bipdeal_walker_heuristic

max_time_steps = 500 * 1000


def run_ddpg(env, env_name):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_ddpg.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'actor_learning_rate',
                   'critic_learning_rate', 'tau', 'assign_priority', 'use_preloaded_memory', 'use_mse',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for batch_size in [64, 128]:
        for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                       64, 128, 256, 512}):
            for actor_learning_rate in [0.001, 0.0001]:
                for critic_learning_rate in [0.001, 0.0001]:
                    for tau in [1e-2, 1e-3]:
                        for assign_priority in [False, True]:
                            for use_preloaded_memory in [False, True]:
                                for use_mse in [False, True]:
                                    actor_optimizer_args = {
                                        'learning_rate': actor_learning_rate
                                    }
                                    critic_optimizer_args = {
                                        'learning_rate': critic_learning_rate
                                    }
                                    network_args = {
                                        'fc_dims': hidden_layer_size
                                    }

                                    agents = {}

                                    for agent_id in env.possible_agents:
                                        if use_preloaded_memory:
                                            pre_loaded_memory = develop_memory_from_pettingzoo_env(env,
                                                                                                   agent_id=agent_id,
                                                                                                   max_time_steps=max_time_steps)
                                        else:
                                            pre_loaded_memory = None
                                        agents.update({agent_id: DDPGAgent(
                                            input_dims=env.observation_space.shape,
                                            action_space=env.action_space,
                                            tau=tau, network_args=network_args,
                                            batch_size=batch_size,
                                            actor_optimizer_type=NetworkOptimizer.ADAM,
                                            critic_optimizer_type=NetworkOptimizer.ADAM,
                                            actor_optimizer_args=actor_optimizer_args,
                                            critic_optimizer_args=critic_optimizer_args,
                                            pre_loaded_memory=pre_loaded_memory,
                                            assign_priority=assign_priority,
                                            use_mse=use_mse
                                        )})

                                    result = run_pettingzoo_env(env, agents, n_games_train=500, n_games_test=50)

                                    new_row = {
                                        'batch_size': batch_size,
                                        'hidden_layer_size': hidden_layer_size,
                                        'actor_learning_rate': actor_learning_rate,
                                        'critic_learning_rate': critic_learning_rate,
                                        'tau': tau,
                                        'assign_priority': 'Yes' if assign_priority else 'No',
                                        'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                        'use_mse': 'Yes' if use_mse else 'No'
                                    }

                                    for key in result:
                                        new_row.update({key: result[key]})

                                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td3(env, env_name):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_td3.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'actor_learning_rate',
                   'critic_learning_rate', 'tau', 'assign_priority', 'use_preloaded_memory', 'use_mse',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    actor_optimizer_args = {
        'learning_rate': 1e-3
    }
    critic_optimizer_args = {
        'learning_rate': 1e-3
    }

    for batch_size in [64, 100, 128]:
        for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                       64, 128, 256, 512}):
            for tau in [0.005, 0.01]:
                for assign_priority in [False, True]:
                    for use_preloaded_memory in [False, True]:
                        for use_mse in [False, True]:
                            network_args = {
                                'fc_dims': hidden_layer_size
                            }

                            agents = {}

                            for agent_id in env.possible_agents:
                                if use_preloaded_memory:
                                    pre_loaded_memory = develop_memory_from_pettingzoo_env(env, agent_id=agent_id,
                                                                                           max_time_steps=max_time_steps)
                                else:
                                    pre_loaded_memory = None
                                agents.update({agent_id: TD3Agent(
                                    input_dims=env.observation_space.shape,
                                    action_space=env.action_space,
                                    tau=tau, network_args=network_args,
                                    batch_size=batch_size,
                                    actor_optimizer_type=NetworkOptimizer.ADAM,
                                    critic_optimizer_type=NetworkOptimizer.ADAM,
                                    actor_optimizer_args=actor_optimizer_args,
                                    critic_optimizer_args=critic_optimizer_args,
                                    assign_priority=assign_priority,
                                    pre_loaded_memory=pre_loaded_memory,
                                    use_mse=use_mse,
                                )})

                            result = run_pettingzoo_env(env, agents, n_games_train=500, n_games_test=50)

                            new_row = {
                                'batch_size': batch_size,
                                'hidden_layer_size': hidden_layer_size,
                                'tau': tau,
                                'assign_priority': 'Yes' if assign_priority else 'No',
                                'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                'use_mse': 'Yes' if use_mse else 'No'
                            }

                            for key in result:
                                new_row.update({key: result[key]})

                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_actor_critic_continuous_methods(env, env_name):
    print('Running', env_name)
    run_ddpg(env, env_name)
    run_td3(env, env_name)


def run_cem(env, env_name):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_cem.csv'.format(env_name))

    result_cols = ['hidden_layer_size',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for hidden_layer_size in list({64, 128, 256, 512}):
        network_args = {
            'fc_dim': hidden_layer_size
        }

        agents = {}

        for agent_id in env.possible_agents:
            agents.update({agent_id: CEMAgent(
                input_dims=env.observation_spaces[agent_id].shape,
                action_space=env.action_spaces[agent_id],
                network_args=network_args
            )})

        new_row = {'hidden_layer_size': hidden_layer_size}

        result = run_pettingzoo_env(env, agents, n_games_train=500, n_games_test=50)

        for key in result:
            new_row.update({key: result[key]})

        results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, float_format='%.3f', index=False)


def run_decision_tree_heuristics(env, env_name, heuristic_func, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_dt.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for learning_type in [LearningType.OFFLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:

            agents = {}

            for agent_id in env.possible_agents:
                agents.update({agent_id: HeuristicWithDT(
                    heuristic_func=heuristic_func, use_model_only=use_model_only,
                    action_space=env.action_spaces[agent_id], input_dims=env.observation_spaces[agent_id].shape,
                    **args
                )})

            result = run_pettingzoo_env(env, agents, learning_type=learning_type, n_games_train=500, n_games_test=50)

            new_row = {
                'learning_type': learning_type.name,
                'use_model_only': 'Yes' if use_model_only else 'No'
            }

            for key in result:
                new_row.update({key: result[key]})

            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, float_format='%.3f', index=False)


def run_random_forest_heuristics(env, env_name, heuristic_func, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_rf.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for learning_type in [LearningType.OFFLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:

            agents = {}
            for agent_id in env.possible_agents:
                agents.update({agent_id: HeuristicWithRF(
                    heuristic_func=heuristic_func, use_model_only=use_model_only,
                    action_space=env.action_spaces[agent_id], input_dims=env.observation_spaces[agent_id].shape,
                    **args
                )})

            result = run_pettingzoo_env(env, agents, learning_type=learning_type, n_games_train=500, n_games_test=50)

            new_row = {
                'learning_type': learning_type.name,
                'use_model_only': 'Yes' if use_model_only else 'No'
            }

            for key in result:
                new_row.update({key: result[key]})

            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, float_format='%.3f', index=False)


def run_ddpg_heuristics(env, env_name, heuristic_func, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristics_ddpg.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size', 'actor_learning_rate',
                   'critic_learning_rate', 'tau', 'use_preloaded_memory', 'use_mse',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for batch_size in [64, 128]:
                for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                               64, 128, 256, 512}):
                    for actor_learning_rate in [0.001, 0.0001]:
                        for critic_learning_rate in [0.001, 0.0001]:
                            for tau in [1e-2, 1e-3]:
                                for use_preloaded_memory in [False, True]:
                                    for use_mse in [False, True]:
                                        actor_optimizer_args = {
                                            'learning_rate': actor_learning_rate
                                        }
                                        critic_optimizer_args = {
                                            'learning_rate': critic_learning_rate
                                        }
                                        network_args = {
                                            'fc_dims': hidden_layer_size
                                        }

                                        agents = {}

                                        for agent_id in env.possible_agents:
                                            pre_loaded_memory = develop_memory_from_pettingzoo_env(env,
                                                                                                   max_time_steps,
                                                                                                   agent_id)
                                            agents.update({agent_id: HeuristicWithDDPG(
                                                input_dims=env.observation_space.shape,
                                                action_space=env.action_space,
                                                tau=tau, network_args=network_args,
                                                batch_size=batch_size,
                                                actor_optimizer_type=NetworkOptimizer.ADAM,
                                                critic_optimizer_type=NetworkOptimizer.ADAM,
                                                actor_optimizer_args=actor_optimizer_args,
                                                critic_optimizer_args=critic_optimizer_args,
                                                learning_type=learning_type,
                                                use_model_only=use_model_only,
                                                heuristic_func=heuristic_func,
                                                use_mse=use_mse,
                                                pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                **args
                                            )})

                                        result = run_pettingzoo_env(env, agents, learning_type=learning_type,
                                                                    n_games_train=500, n_games_test=50)

                                        new_row = {
                                            'learning_type': learning_type.name,
                                            'use_model_only': 'Yes' if use_model_only else 'No',
                                            'batch_size': batch_size,
                                            'hidden_layer_size': hidden_layer_size,
                                            'actor_learning_rate': actor_learning_rate,
                                            'critic_learning_rate': critic_learning_rate,
                                            'tau': tau,
                                            'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                            'use_mse': 'Yes' if use_mse else 'No'
                                        }

                                        for key in result:
                                            new_row.update({key: result[key]})

                                        results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td3_heuristics(env, env_name, heuristic_func, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristics_td3.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size', 'actor_learning_rate',
                   'critic_learning_rate', 'tau', 'use_preloaded_memory', 'use_mse',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for batch_size in [64, 128]:
                for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                               64, 128, 256, 512}):
                    for actor_learning_rate in [0.001, 0.0001]:
                        for critic_learning_rate in [0.001, 0.0001]:
                            for tau in [1e-2, 1e-3]:
                                for use_preloaded_memory in [False, True]:
                                    for use_mse in [False, True]:
                                        actor_optimizer_args = {
                                            'learning_rate': actor_learning_rate
                                        }
                                        critic_optimizer_args = {
                                            'learning_rate': critic_learning_rate
                                        }
                                        network_args = {
                                            'fc_dims': hidden_layer_size
                                        }

                                        agents = {}
                                        for agent_id in env.possible_agents:
                                            pre_loaded_memory = develop_memory_from_pettingzoo_env(env, max_time_steps,
                                                                                                   agent_id)

                                            agents.update({agent_id: HeuristicWithTD3(
                                                input_dims=env.observation_space.shape,
                                                action_space=env.action_space,
                                                tau=tau, network_args=network_args,
                                                batch_size=batch_size,
                                                actor_optimizer_type=NetworkOptimizer.ADAM,
                                                critic_optimizer_type=NetworkOptimizer.ADAM,
                                                actor_optimizer_args=actor_optimizer_args,
                                                critic_optimizer_args=critic_optimizer_args,
                                                learning_type=learning_type,
                                                use_model_only=use_model_only,
                                                heuristic_func=heuristic_func,
                                                use_mse=use_mse,
                                                pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                **args
                                            )})

                                        result = run_pettingzoo_env(env, agents, learning_type=learning_type,
                                                                    n_games_train=500, n_games_test=50)

                                        new_row = {
                                            'learning_type': learning_type.name,
                                            'use_model_only': 'Yes' if use_model_only else 'No',
                                            'batch_size': batch_size,
                                            'hidden_layer_size': hidden_layer_size,
                                            'actor_learning_rate': actor_learning_rate,
                                            'critic_learning_rate': critic_learning_rate,
                                            'tau': tau,
                                            'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                            'use_mse': 'Yes' if use_mse else 'No'
                                        }

                                        for key in result:
                                            new_row.update({key: result[key]})

                                        results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_cem_heuristics(env, env_name, heuristic_func, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_cem.csv'.format(env_name))

    result_cols = ['use_model_only', 'learning_type', 'hidden_layer_size',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for use_model_only in [False, True]:
        for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
            for hidden_layer_size in list({64, 128, 256, 512}):
                network_args = {
                    'fc_dim': hidden_layer_size
                }
                agent = HeuristicWithCEM(input_dims=env.observation_space.shape, action_space=env.action_shape,
                                         network_args=network_args, heuristic_func=heuristic_func,
                                         use_model_only=use_model_only,
                                         **args)

                new_row = {'hidden_layer_size': hidden_layer_size, 'learning_type': learning_type,
                           'use_model_only': 'Yes' if use_model_only else 'No'}

                agents = {}
                for agent_id in env.possible_agents:
                    agents.update({agent_id: agent})

                result = run_pettingzoo_env(env, agents, n_games_train=500, n_games_test=50,
                                            learning_type=learning_type)

                for key in result:
                    new_row.update({key: result[key]})

                results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, float_format='%.3f', index=False)


def run_heuristics(env, env_name, heuristic_func, **args):
    run_decision_tree_heuristics(env, env_name, heuristic_func, **args)
    run_random_forest_heuristics(env, env_name, heuristic_func, **args)
    run_ddpg_heuristics(env, env_name, heuristic_func, **args)
    run_td3_heuristics(env, env_name, heuristic_func, **args)
    run_cem_heuristics(env, env_name, heuristic_func, **args)


env = multiwalker_v7.env()
run_actor_critic_continuous_methods(env, 'multiwalker')
run_heuristics(env, 'multiwalker', bipdeal_walker_heuristic, state=1, moving_leg=0, supporting_leg=1,
               supporting_knee_angle=0.1, min_supporting_knee_angle=0.1)
