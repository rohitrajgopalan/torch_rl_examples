import os

import numpy as np
import pandas as pd
from pettingzoo.classic import tictactoe_v3
from torch_rl.dueling_td.agent import DuelingTDAgent
from torch_rl.heuristic.heuristic_with_dt import HeuristicWithDT
from torch_rl.heuristic.heuristic_with_dueling_td import HeuristicWithDuelingTD
from torch_rl.heuristic.heuristic_with_hill_climbing import HeuristicWithHillClimbing
from torch_rl.heuristic.heuristic_with_rf import HeuristicWithRF
from torch_rl.heuristic.heuristic_with_td import HeuristicWithTD
from torch_rl.hill_climbing.agent import HillClimbingAgent
from torch_rl.td.agent import TDAgent
from torch_rl.utils.types import NetworkOptimizer, TDAlgorithmType, PolicyType, LearningType

from common.run import run_pettingzoo_env
from common.utils import develop_memory_from_pettingzoo_env
from random_legal import RandomLegal
from tic_tac_toe_heuristic import TicTacToeHeuristic

max_time_steps = 500 * 1000


def run_td_epsilon_greedy(env, env_name, penalty):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_td_epsilon_greedy.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate', 'goal_focused',
                   'assign_priority', 'enable_action_blocker', 'action_blocker_model_type', 'use_preloaded_memory',
                   'is_double', 'algorithm_type', 'enable_decay', 'epsilon_start', 'use_mse', 'opponent',
                   'player_number',
                   'learning_type', 'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in list({False, penalty > 0}):
                action_blocker_model_types = ['decision_tree', 'random_forest'] if enable_action_blocker else [None]
                for action_blocker_model_type in action_blocker_model_types:
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                for hidden_layer_size in [64, 128, 256, 512]:
                                    for enable_decay in [False, True]:
                                        epsilons = [1.0] if enable_decay \
                                            else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                                        for epsilon in epsilons:
                                            policy_args.update({'eps_start': epsilon, 'enable_decay': enable_decay})
                                            for assign_priority in [False, True]:
                                                for use_preloaded_memory in [False, True]:
                                                    for use_mse in [False, True]:
                                                        for learning_type in [LearningType.OFFLINE, LearningType.ONLINE,
                                                                              LearningType.BOTH]:
                                                            for opponent in ['heuristic', 'random_legal']:
                                                                for player_number in [1, 2]:
                                                                    opponent_id = (2 - player_number) + 1
                                                                    network_optimizer_args = {
                                                                        'learning_rate': learning_rate
                                                                    }
                                                                    network_args = {
                                                                        'fc_dims': hidden_layer_size
                                                                    }
                                                                    pre_loaded_memory = develop_memory_from_pettingzoo_env(
                                                                        env, max_time_steps,
                                                                        'player_{0}'.format(player_number))
                                                                    agent = TDAgent(
                                                                        input_dims=env.observation_space.shape,
                                                                        action_space=env.action_space,
                                                                        gamma=0.99,
                                                                        mem_size=1000,
                                                                        batch_size=batch_size,
                                                                        network_args=network_args,
                                                                        optimizer_type=optimizer_type,
                                                                        replace=1000,
                                                                        optimizer_args=network_optimizer_args,
                                                                        enable_action_blocking=enable_action_blocker,
                                                                        min_penalty=penalty,
                                                                        is_double=is_double,
                                                                        algorithm_type=algorithm_type,
                                                                        policy_type=PolicyType.EPSILON_GREEDY,
                                                                        policy_args=policy_args,
                                                                        assign_priority=assign_priority,
                                                                        pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                                        action_blocker_timesteps=max_time_steps,
                                                                        action_blocker_model_type=action_blocker_model_type,
                                                                        use_mse=use_mse)

                                                                    agents = {
                                                                        'player_{0}'.format(
                                                                            opponent_id): TicTacToeHeuristic(
                                                                            'player_{0}'.format(
                                                                                opponent_id)) if opponent == 'heuristic' else RandomLegal(),
                                                                        'player_{0}'.format(player_number): agent
                                                                    }
                                                                    result = run_pettingzoo_env(env, agents,
                                                                                                n_games_train=500,
                                                                                                n_games_test=50,
                                                                                                learning_type=learning_type)

                                                                    new_row = {
                                                                        'batch_size': batch_size,
                                                                        'hidden_layer_size': hidden_layer_size,
                                                                        'algorithm_type': algorithm_type,
                                                                        'optimizer': optimizer_type.name.lower(),
                                                                        'learning_rate': learning_rate,
                                                                        'opponent': opponent,
                                                                        'player_number': player_number,
                                                                        'learning_type': learning_type.name,
                                                                        'is_double': 'Yes' if is_double else 'No',
                                                                        'enable_decay': 'Yes' if enable_decay else 'No',
                                                                        'epsilon': epsilon,
                                                                        'assign_priority': 'Yes' if assign_priority else 'No',
                                                                        'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                                                        'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                                                        'action_blocker_model_type': action_blocker_model_type if enable_action_blocker else 'N/A',
                                                                        'use_mse': 'Yes' if use_mse else 'No'
                                                                    }
                                                                    for key in result:
                                                                        new_row.update({key: result[key]})

                                                                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td_softmax(env, env_name, penalty):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_td_softmax.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate', 'goal_focused',
                   'assign_priority', 'enable_action_blocker', 'action_blocker_model_type', 'use_preloaded_memory',
                   'is_double', 'algorithm_type', 'tau', 'use_mse', 'opponent', 'player_number', 'learning_type',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)
    policy_args = {}

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in list({False, penalty > 0}):
                action_blocker_model_types = ['decision_tree', 'random_forest'] if enable_action_blocker else [None]
                for action_blocker_model_type in action_blocker_model_types:
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                for hidden_layer_size in [64, 128, 256, 512]:
                                    for tau in [0.0001, 0.001, 0.1, 1.0, 10.0]:
                                        policy_args.update({'tau': tau})
                                        for assign_priority in [False, True]:
                                            for use_preloaded_memory in [False, True]:
                                                for use_mse in [False, True]:
                                                    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE,
                                                                          LearningType.BOTH]:
                                                        for opponent in ['heuristic', 'random_legal']:
                                                            for player_number in [1, 2]:
                                                                opponent_id = (2 - player_number) + 1
                                                                network_optimizer_args = {
                                                                    'learning_rate': learning_rate
                                                                }
                                                                network_args = {
                                                                    'fc_dims': hidden_layer_size
                                                                }
                                                                pre_loaded_memory = develop_memory_from_pettingzoo_env(
                                                                    env, max_time_steps,
                                                                    'player_{0}'.format(player_number))
                                                                agent = TDAgent(
                                                                    input_dims=env.observation_space.shape,
                                                                    action_space=env.action_space,
                                                                    gamma=0.99,
                                                                    mem_size=1000,
                                                                    batch_size=batch_size,
                                                                    network_args=network_args,
                                                                    optimizer_type=optimizer_type,
                                                                    replace=1000,
                                                                    optimizer_args=network_optimizer_args,
                                                                    enable_action_blocking=enable_action_blocker,
                                                                    min_penalty=penalty,
                                                                    is_double=is_double,
                                                                    algorithm_type=algorithm_type,
                                                                    policy_type=PolicyType.SOFTMAX,
                                                                    policy_args=policy_args,
                                                                    assign_priority=assign_priority,
                                                                    pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                                    action_blocker_model_type=action_blocker_model_type,
                                                                    action_blocker_timesteps=max_time_steps,
                                                                    use_mse=use_mse)

                                                                agents = {
                                                                    'player_{0}'.format(
                                                                        opponent_id): TicTacToeHeuristic(
                                                                        'player_{0}'.format(
                                                                            opponent_id)) if opponent == 'heuristic' else RandomLegal(),
                                                                    'player_{0}'.format(player_number): agent
                                                                }

                                                                result = run_pettingzoo_env(env, agents,
                                                                                            n_games_train=500,
                                                                                            n_games_test=50,
                                                                                            learning_type=learning_type)

                                                                new_row = {
                                                                    'batch_size': batch_size,
                                                                    'hidden_layer_size': hidden_layer_size,
                                                                    'algorithm_type': algorithm_type,
                                                                    'optimizer': optimizer_type.name.lower(),
                                                                    'learning_rate': learning_rate,
                                                                    'is_double': 'Yes' if is_double else 'No',
                                                                    'tau': tau,
                                                                    'assign_priority': 'Yes' if assign_priority else 'No',
                                                                    'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                                                    'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                                                    'action_blocker_model_type': action_blocker_model_type if enable_action_blocker else 'N/A',
                                                                    'use_mse': 'Yes' if use_mse else 'No',
                                                                    'opponent': opponent,
                                                                    'player_number': player_number,
                                                                    'learning_type': learning_type.name
                                                                }
                                                                for key in result:
                                                                    new_row.update({key: result[key]})

                                                                results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td_ucb(env, env_name, penalty):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results', '{0}_td_ucb.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate', 'goal_focused',
                   'assign_priority', 'is_double', 'algorithm_type',
                   'enable_action_blocker', 'action_blocker_model_type', 'use_preloaded_memory', 'use_mse',
                   'opponent', 'player_number', 'learning_type',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {'confidence_factor': 2}

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in list({False, penalty > 0}):
                action_blocker_model_types = ['decision_tree', 'random_forest'] if enable_action_blocker else [None]
                for action_blocker_model_type in action_blocker_model_types:
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                for hidden_layer_size in [64, 128, 256, 512]:
                                    for assign_priority in [False, True]:
                                        for use_preloaded_memory in [False, True]:
                                            for use_mse in [False, True]:
                                                for learning_type in [LearningType.OFFLINE, LearningType.ONLINE,
                                                                      LearningType.BOTH]:
                                                    for opponent in ['heuristic', 'random_legal']:
                                                        for player_number in [1, 2]:
                                                            opponent_id = (2 - player_number) + 1
                                                            pre_loaded_memory = develop_memory_from_pettingzoo_env(env,
                                                                                                                   max_time_steps,
                                                                                                                   'player_{0}'.format(
                                                                                                                       player_number))
                                                            network_optimizer_args = {
                                                                'learning_rate': learning_rate
                                                            }
                                                            network_args = {
                                                                'fc_dims': hidden_layer_size
                                                            }
                                                            agent = TDAgent(
                                                                input_dims=env.observation_space.shape,
                                                                action_space=env.action_space,
                                                                gamma=0.99,
                                                                mem_size=1000,
                                                                batch_size=batch_size,
                                                                network_args=network_args,
                                                                optimizer_type=optimizer_type,
                                                                replace=1000,
                                                                optimizer_args=network_optimizer_args,
                                                                enable_action_blocking=enable_action_blocker,
                                                                min_penalty=penalty,
                                                                is_double=is_double,
                                                                algorithm_type=algorithm_type,
                                                                policy_type=PolicyType.UCB,
                                                                policy_args=policy_args,
                                                                assign_priority=assign_priority,
                                                                pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                                action_blocker_model_type=action_blocker_model_type,
                                                                action_blocker_timesteps=max_time_steps,
                                                                use_mse=use_mse
                                                            )

                                                            agents = {
                                                                'player_{0}'.format(opponent_id): TicTacToeHeuristic(
                                                                    'player_{0}'.format(
                                                                        opponent_id)) if opponent == 'heuristic' else RandomLegal(),
                                                                'player_{0}'.format(player_number): agent
                                                            }

                                                            result = run_pettingzoo_env(env, agents, n_games_train=500,
                                                                                        n_games_test=50,
                                                                                        learning_type=learning_type)

                                                            new_row = {
                                                                'batch_size': batch_size,
                                                                'hidden_layer_size': hidden_layer_size,
                                                                'algorithm_type': algorithm_type,
                                                                'optimizer': optimizer_type.name.lower(),
                                                                'learning_rate': learning_rate,
                                                                'opponent': opponent,
                                                                'player_number': player_number,
                                                                'learning_type': learning_type.name,
                                                                'is_double': 'Yes' if is_double else 'No',
                                                                'assign_priority': 'Yes' if assign_priority else 'No',
                                                                'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                                                'use_preloaded_memory': 'Yes' if use_preloaded_memory else "No",
                                                                'action_blocker_model_type': action_blocker_model_type if enable_action_blocker else 'N/A',
                                                                'use_mse': 'Yes' if use_mse else 'No'
                                                            }
                                                            for key in result:
                                                                new_row.update({key: result[key]})

                                                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td_thompson_sampling(env, env_name, penalty):
    assert penalty > 0

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_td_thompson_sampling.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate', 'goal_focused',
                   'assign_priority', 'is_double', 'algorithm_type',
                   'enable_action_blocker', 'action_blocker_model_type', 'use_preloaded_memory', 'use_mse',
                   'opponent', 'player_number', 'learning_type',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {
        'min_penalty': penalty
    }

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in list({False, penalty > 0}):
                action_blocker_model_types = ['decision_tree', 'random_forest'] if enable_action_blocker else [None]
                for action_blocker_model_type in action_blocker_model_types:
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                for hidden_layer_size in [64, 128, 256, 512]:
                                    for assign_priority in [False, True]:
                                        for use_preloaded_memory in [False, True]:
                                            for use_mse in [False, True]:
                                                for learning_type in [LearningType.OFFLINE, LearningType.ONLINE,
                                                                      LearningType.BOTH]:
                                                    for opponent in ['heuristic', 'random_legal']:
                                                        for player_number in [1, 2]:
                                                            opponent_id = (2 - player_number) + 1
                                                            pre_loaded_memory = develop_memory_from_pettingzoo_env(env,
                                                                                                                   max_time_steps,
                                                                                                                   'player_{0}'.format(
                                                                                                                       player_number))
                                                            network_optimizer_args = {
                                                                'learning_rate': learning_rate
                                                            }
                                                            network_args = {
                                                                'fc_dims': hidden_layer_size
                                                            }
                                                            agent = TDAgent(
                                                                input_dims=env.observation_space.shape,
                                                                action_space=env.action_space,
                                                                gamma=0.99,
                                                                mem_size=1000,
                                                                batch_size=batch_size,
                                                                network_args=network_args,
                                                                optimizer_type=optimizer_type,
                                                                replace=1000,
                                                                optimizer_args=network_optimizer_args,
                                                                enable_action_blocking=enable_action_blocker,
                                                                min_penalty=penalty,
                                                                is_double=is_double,
                                                                algorithm_type=algorithm_type,
                                                                policy_type=PolicyType.THOMPSON_SAMPLING,
                                                                policy_args=policy_args,
                                                                assign_priority=assign_priority,
                                                                action_blocker_model_type=action_blocker_model_type,
                                                                action_blocker_timesteps=max_time_steps,
                                                                pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                                use_mse=use_mse)

                                                            agents = {
                                                                'player_{0}'.format(opponent_id): TicTacToeHeuristic(
                                                                    'player_{0}'.format(
                                                                        opponent_id)) if opponent == 'heuristic' else RandomLegal(),
                                                                'player_{0}'.format(player_number): agent
                                                            }

                                                            result = run_pettingzoo_env(env, agents, n_games_train=500,
                                                                                        n_games_test=50,
                                                                                        learning_type=learning_type)

                                                            new_row = {
                                                                'batch_size': batch_size,
                                                                'hidden_layer_size': hidden_layer_size,
                                                                'algorithm_type': algorithm_type,
                                                                'optimizer': optimizer_type.name.lower(),
                                                                'learning_rate': learning_rate,
                                                                'opponent': opponent,
                                                                'player_number': player_number,
                                                                'learning_type': learning_type.name,
                                                                'is_double': 'Yes' if is_double else 'No',
                                                                'assign_priority': 'Yes' if assign_priority else 'No',
                                                                'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                                                'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                                                'action_blocker_model_type': action_blocker_model_type if enable_action_blocker else 'N/A',
                                                                'use_mse': 'Yes' if use_mse else 'No'
                                                            }
                                                            for key in result:
                                                                new_row.update({key: result[key]})

                                                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_epsilon_greedy(env, env_name, penalty):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_td_epsilon_greedy.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate', 'goal_focused',
                   'is_double', 'algorithm_type', 'enable_decay',
                   'epsilon_start', 'enable_action_blocker', 'action_blocker_model_type', 'use_preloaded_memory',
                   'opponent', 'player_number', 'learning_type',
                   'use_mse', 'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in list({False, penalty > 0}):
                action_blocker_model_types = ['decision_tree', 'random_forest'] if enable_action_blocker else [None]
                for action_blocker_model_type in action_blocker_model_types:
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                for hidden_layer_size in [64, 128, 256, 512]:
                                    for enable_decay in [False, True]:
                                        epsilons = [1.0] if enable_decay \
                                            else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                                        for epsilon in epsilons:
                                            policy_args.update({'eps_start': epsilon, 'enable_decay': enable_decay})
                                            for assign_priority in [False, True]:
                                                for use_preloaded_memory in [False, True]:
                                                    for use_mse in [False, True]:
                                                        for learning_type in [LearningType.OFFLINE, LearningType.ONLINE,
                                                                              LearningType.BOTH]:
                                                            for opponent in ['heuristic', 'random_legal']:
                                                                for player_number in [1, 2]:
                                                                    opponent_id = (2 - player_number) + 1
                                                                    pre_loaded_memory = develop_memory_from_pettingzoo_env(
                                                                        env, max_time_steps,
                                                                        'player_{0}'.format(player_number))
                                                                    network_optimizer_args = {
                                                                        'learning_rate': learning_rate
                                                                    }
                                                                    network_args = {
                                                                        'fc_dims': hidden_layer_size
                                                                    }
                                                                    agent = DuelingTDAgent(
                                                                        input_dims=env.observation_space.shape,
                                                                        action_space=env.action_space,
                                                                        gamma=0.99,
                                                                        mem_size=1000,
                                                                        batch_size=batch_size,
                                                                        network_args=network_args,
                                                                        optimizer_type=optimizer_type,
                                                                        replace=1000,
                                                                        optimizer_args=network_optimizer_args,
                                                                        enable_action_blocking=enable_action_blocker,
                                                                        min_penalty=penalty,
                                                                        is_double=is_double,
                                                                        algorithm_type=algorithm_type,
                                                                        policy_type=PolicyType.EPSILON_GREEDY,
                                                                        policy_args=policy_args,
                                                                        assign_priority=assign_priority,
                                                                        action_blocker_model_type=action_blocker_model_type,
                                                                        action_blocker_timesteps=max_time_steps,
                                                                        pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                                        use_mse=use_mse)

                                                                    agents = {
                                                                        'player_{0}'.format(
                                                                            opponent_id): TicTacToeHeuristic(
                                                                            'player_{0}'.format(
                                                                                opponent_id)) if opponent == 'heuristic' else RandomLegal(),
                                                                        'player_{0}'.format(player_number): agent
                                                                    }
                                                                    result = run_pettingzoo_env(env, agents,
                                                                                                n_games_train=500,
                                                                                                n_games_test=50,
                                                                                                learning_type=learning_type)

                                                                    new_row = {
                                                                        'batch_size': batch_size,
                                                                        'hidden_layer_size': hidden_layer_size,
                                                                        'algorithm_type': algorithm_type,
                                                                        'optimizer': optimizer_type.name.lower(),
                                                                        'learning_rate': learning_rate,
                                                                        'opponent': opponent,
                                                                        'player_number': player_number,
                                                                        'learning_type': learning_type.name,
                                                                        'is_double': 'Yes' if is_double else 'No',
                                                                        'enable_decay': 'Yes' if enable_decay else 'No',
                                                                        'epsilon': epsilon,
                                                                        'assign_priority': 'Yes' if assign_priority else 'No',
                                                                        'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                                                        'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                                                        'action_blocker_model_type': action_blocker_model_type if enable_action_blocker else 'N/A',
                                                                        'use_mse': 'Yes' if use_mse else 'No'
                                                                    }
                                                                    for key in result:
                                                                        new_row.update({key: result[key]})

                                                                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_softmax(env, env_name, penalty):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_td_softmax.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate', 'goal_focused',
                   'assign_priority', 'is_double', 'algorithm_type', 'tau',
                   'enable_action_blocker', 'action_blocker_model_type', 'use_preloaded_memory', 'use_mse',
                   'opponent', 'player_number', 'learning_type',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in list({False, penalty > 0}):
                action_blocker_model_types = ['decision_tree', 'random_forest'] if enable_action_blocker else [None]
                for action_blocker_model_type in action_blocker_model_types:
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                for hidden_layer_size in [64, 128, 256, 512]:
                                    for tau in [0.0001, 0.001, 0.1, 1.0, 10.0]:
                                        policy_args.update({'tau': tau})
                                        for assign_priority in [False, True]:
                                            for use_preloaded_memory in [False, True]:
                                                for use_mse in [False, True]:
                                                    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE,
                                                                          LearningType.BOTH]:
                                                        for opponent in ['heuristic', 'random_legal']:
                                                            for player_number in [1, 2]:
                                                                opponent_id = (2 - player_number) + 1
                                                                pre_loaded_memory = develop_memory_from_pettingzoo_env(
                                                                    env, max_time_steps,
                                                                    'player_{0}'.format(player_number))
                                                                network_optimizer_args = {
                                                                    'learning_rate': learning_rate
                                                                }
                                                                network_args = {
                                                                    'fc_dims': hidden_layer_size
                                                                }
                                                                agent = DuelingTDAgent(
                                                                    input_dims=env.observation_space.shape,
                                                                    action_space=env.action_space,
                                                                    gamma=0.99,
                                                                    mem_size=1000,
                                                                    batch_size=batch_size,
                                                                    network_args=network_args,
                                                                    optimizer_type=optimizer_type,
                                                                    replace=1000,
                                                                    optimizer_args=network_optimizer_args,
                                                                    enable_action_blocking=enable_action_blocker,
                                                                    min_penalty=penalty,
                                                                    is_double=is_double,
                                                                    algorithm_type=algorithm_type,
                                                                    policy_type=PolicyType.SOFTMAX,
                                                                    policy_args=policy_args,
                                                                    assign_priority=assign_priority,
                                                                    action_blocker_model_type=action_blocker_model_type,
                                                                    action_blocker_timesteps=max_time_steps,
                                                                    pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                                    use_mse=use_mse)

                                                                agents = {
                                                                    'player_{0}'.format(
                                                                        opponent_id): TicTacToeHeuristic(
                                                                        'player_{0}'.format(
                                                                            opponent_id)) if opponent == 'heuristic' else RandomLegal(),
                                                                    'player_{0}'.format(player_number): agent
                                                                }

                                                                result = run_pettingzoo_env(env, agents,
                                                                                            n_games_train=500,
                                                                                            n_games_test=50,
                                                                                            learning_type=learning_type)

                                                                new_row = {
                                                                    'batch_size': batch_size,
                                                                    'hidden_layer_size': hidden_layer_size,
                                                                    'algorithm_type': algorithm_type,
                                                                    'optimizer': optimizer_type.name.lower(),
                                                                    'learning_rate': learning_rate,
                                                                    'opponent': opponent,
                                                                    'player_number': player_number,
                                                                    'learning_type': learning_type.name,
                                                                    'is_double': 'Yes' if is_double else 'No',
                                                                    'tau': tau,
                                                                    'assign_priority': 'Yes' if assign_priority else 'No',
                                                                    'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                                                    'action_blocker_model_type': action_blocker_model_type if enable_action_blocker else 'N/A',
                                                                    'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                                                    'use_mse': 'Yes' if use_mse else 'No'
                                                                }
                                                                for key in result:
                                                                    new_row.update({key: result[key]})

                                                                results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_ucb(env, env_name, penalty):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_td_ucb.csv'.format(env_name))
    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate', 'goal_focused',
                   'assign_priority', 'is_double', 'algorithm_type',
                   'enable_action_blocker', 'action_blocker_model_type', 'use_preloaded_memory', 'use_mse',
                   'opponent', 'player_number', 'learning_type',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {'confidence_factor': 2}

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in list({False, penalty > 0}):
                action_blocker_model_types = ['decision_tree', 'random_forest'] if enable_action_blocker else [None]
                for action_blocker_model_type in action_blocker_model_types:
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                for hidden_layer_size in [64, 128, 256, 512]:
                                    for assign_priority in [False, True]:
                                        for use_preloaded_memory in [False, True]:
                                            for use_mse in [False, True]:
                                                for learning_type in [LearningType.OFFLINE, LearningType.ONLINE,
                                                                      LearningType.BOTH]:
                                                    for opponent in ['heuristic', 'random_legal']:
                                                        for player_number in [1, 2]:
                                                            opponent_id = (2 - player_number) + 1
                                                            pre_loaded_memory = develop_memory_from_pettingzoo_env(env,
                                                                                                                   max_time_steps,
                                                                                                                   'player_{0}'.format(
                                                                                                                       player_number))
                                                            network_optimizer_args = {
                                                                'learning_rate': learning_rate
                                                            }
                                                            network_args = {
                                                                'fc_dims': hidden_layer_size
                                                            }
                                                            agent = DuelingTDAgent(
                                                                input_dims=env.observation_space.shape,
                                                                action_space=env.action_space,
                                                                gamma=0.99,
                                                                mem_size=1000,
                                                                batch_size=batch_size,
                                                                network_args=network_args,
                                                                optimizer_type=optimizer_type,
                                                                replace=1000,
                                                                optimizer_args=network_optimizer_args,
                                                                enable_action_blocking=enable_action_blocker,
                                                                min_penalty=penalty,
                                                                is_double=is_double,
                                                                algorithm_type=algorithm_type,
                                                                policy_type=PolicyType.UCB,
                                                                policy_args=policy_args,
                                                                assign_priority=assign_priority,
                                                                action_blocker_model_type=action_blocker_model_type,
                                                                action_blocker_timesteps=max_time_steps,
                                                                pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                                use_mse=use_mse)

                                                            agents = {
                                                                'player_{0}'.format(opponent_id): TicTacToeHeuristic(
                                                                    'player_{0}'.format(
                                                                        opponent_id)) if opponent == 'heuristic' else RandomLegal(),
                                                                'player_{0}'.format(player_number): agent
                                                            }

                                                            new_row = {
                                                                'batch_size': batch_size,
                                                                'hidden_layer_size': hidden_layer_size,
                                                                'algorithm_type': algorithm_type,
                                                                'optimizer': optimizer_type.name.lower(),
                                                                'learning_rate': learning_rate,
                                                                'opponent': opponent,
                                                                'player_number': player_number,
                                                                'learning_type': learning_type.name,
                                                                'is_double': 'Yes' if is_double else 'No',
                                                                'assign_priority': 'Yes' if assign_priority else 'No',
                                                                'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                                                'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                                                'action_blocker_model_type': action_blocker_model_type if enable_action_blocker else 'N/A',
                                                                'use_mse': 'Yes' if use_mse else 'No'
                                                            }

                                                            result = run_pettingzoo_env(env, agents, n_games_train=500,
                                                                                        n_games_test=50,
                                                                                        learning_type=learning_type)

                                                            for key in result:
                                                                new_row.update({key: result[key]})

                                                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_thompson_sampling(env, env_name, penalty):
    assert penalty > 0
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_td_thompson_sampling.csv'.format(env_name))
    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate', 'goal_focused',
                   'assign_priority', 'is_double', 'algorithm_type',
                   'enable_action_blocker', 'action_blocker_model_type', 'use_preloaded_memory', 'use_mse',
                   'opponent', 'player_number', 'learning_type',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {
        'min_penalty': penalty
    }

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in list({False, penalty > 0}):
                action_blocker_model_types = ['decision_tree', 'random_forest'] if enable_action_blocker else [None]
                for action_blocker_model_type in action_blocker_model_types:
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                for hidden_layer_size in [64, 128, 256, 512]:
                                    for assign_priority in [False, True]:
                                        for use_preloaded_memory in [False, True]:
                                            for use_mse in [False, True]:
                                                for learning_type in [LearningType.OFFLINE, LearningType.ONLINE,
                                                                      LearningType.BOTH]:
                                                    for opponent in ['heuristic', 'random_legal']:
                                                        for player_number in [1, 2]:
                                                            opponent_id = (2 - player_number) + 1
                                                            pre_loaded_memory = develop_memory_from_pettingzoo_env(env,
                                                                                                                   max_time_steps,
                                                                                                                   'player_{0}'.format(
                                                                                                                       player_number))
                                                            network_optimizer_args = {
                                                                'learning_rate': learning_rate
                                                            }
                                                            network_args = {
                                                                'fc_dims': hidden_layer_size
                                                            }
                                                            agent = DuelingTDAgent(
                                                                input_dims=env.observation_space.shape,
                                                                action_space=env.action_space,
                                                                gamma=0.99,
                                                                mem_size=1000,
                                                                batch_size=batch_size,
                                                                network_args=network_args,
                                                                optimizer_type=optimizer_type,
                                                                replace=1000,
                                                                optimizer_args=network_optimizer_args,
                                                                enable_action_blocking=enable_action_blocker,
                                                                min_penalty=penalty,
                                                                is_double=is_double,
                                                                algorithm_type=algorithm_type,
                                                                policy_type=PolicyType.THOMPSON_SAMPLING,
                                                                policy_args=policy_args,
                                                                assign_priority=assign_priority,
                                                                action_blocker_model_type=action_blocker_model_type,
                                                                action_blocker_timesteps=max_time_steps,
                                                                pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                                use_mse=use_mse)

                                                            agents = {
                                                                'player_{0}'.format(opponent_id): TicTacToeHeuristic(
                                                                    'player_{0}'.format(
                                                                        opponent_id)) if opponent == 'heuristic' else RandomLegal(),
                                                                'player_{0}'.format(player_number): agent
                                                            }

                                                            result = run_pettingzoo_env(env, agents, n_games_train=500,
                                                                                        n_games_test=50,
                                                                                        learning_type=learning_type)

                                                            new_row = {
                                                                'batch_size': batch_size,
                                                                'hidden_layer_size': hidden_layer_size,
                                                                'algorithm_type': algorithm_type,
                                                                'optimizer': optimizer_type.name.lower(),
                                                                'learning_rate': learning_rate,
                                                                'opponent': opponent,
                                                                'player_number': player_number,
                                                                'learning_type': learning_type.name,
                                                                'is_double': 'Yes' if is_double else 'No',
                                                                'assign_priority': 'Yes' if assign_priority else 'No',
                                                                'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                                                'action_blocker_model_type': action_blocker_model_type if enable_action_blocker else 'N/A',
                                                                'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                                                'use_mse': 'Yes' if use_mse else 'No'
                                                            }
                                                            for key in result:
                                                                new_row.update({key: result[key]})

                                                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_all_td_methods(env, env_name, penalty):
    print('Running', env_name)
    run_td_epsilon_greedy(env, env_name, penalty)
    run_dueling_td_epsilon_greedy(env, env_name, penalty)
    run_td_softmax(env, env_name, penalty)
    run_dueling_td_softmax(env, env_name, penalty)
    run_td_ucb(env, env_name, penalty)
    run_dueling_td_ucb(env, env_name, penalty)
    if penalty > 0:
        run_td_thompson_sampling(env, env_name, penalty)
        run_dueling_td_thompson_sampling(env, env_name, penalty)


def run_hill_climbing(env, env_name, penalty):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_hill_climbing.csv'.format(env_name))
    result_cols = ['enable_action_blocker', 'use_preloaded_memory', 'opponent', 'player_number', 'learning_type',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for enable_action_blocker in list({False, penalty > 0}):
        action_blocker_model_types = ['decision_tree', 'random_forest'] if enable_action_blocker else [None]
        for action_blocker_model_type in action_blocker_model_types:
            for use_preloaded_memory in [False, True]:
                for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
                    for opponent in ['heuristic', 'random_legal']:
                        for player_number in [1, 2]:
                            opponent_id = (2 - player_number) + 1
                            pre_loaded_memory = develop_memory_from_pettingzoo_env(env, max_time_steps,
                                                                                   'player_{0}'.format(player_number))
                            agent = HillClimbingAgent(input_dims=env.observation_space.shape,
                                                      action_space=env.action_space,
                                                      enable_action_blocking=enable_action_blocker,
                                                      action_blocker_model_type=action_blocker_model_type,
                                                      action_blocker_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                      action_blocker_timesteps=max_time_steps,
                                                      gamma=1.0)

                            agents = {
                                'player_{0}'.format(opponent_id): TicTacToeHeuristic(
                                    'player_{0}'.format(opponent_id)) if opponent == 'heuristic' else RandomLegal(),
                                'player_{0}'.format(player_number): agent
                            }

                            new_row = {'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                       'opponent': opponent,
                                       'player_number': player_number,
                                       'learning_type': learning_type.name
                                       }

                            result = run_pettingzoo_env(env, agents, n_games_train=500, n_games_test=50,
                                                        learning_type=learning_type)

                            for key in result:
                                new_row.update({key: result[key]})

                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_decision_tree_heuristics(env, env_name, heuristic_func, min_penalty=0, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_dt.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'enable_action_blocking',
                   'opponent', 'player_number',
                   'use_preloaded_memory', 'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for learning_type in [LearningType.OFFLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for enable_action_blocking in list({False, min_penalty > 0}):
                for use_preloaded_memory in [False, True]:
                    for opponent in ['heuristic', 'random_legal']:
                        for player_number in [1, 2]:
                            opponent_id = (2 - player_number) + 1
                            pre_loaded_memory = develop_memory_from_pettingzoo_env(env, max_time_steps,
                                                                                   'player_{0}'.format(
                                                                                       player_number))
                            agent = HeuristicWithDT(env.observation_space.shape, heuristic_func, use_model_only,
                                                    env.action_space,
                                                    enable_action_blocking, min_penalty,
                                                    pre_loaded_memory if use_preloaded_memory else None,
                                                    None, None, max_time_steps, 'decision_tree', **args)

                            agents = {
                                'player_{0}'.format(opponent_id): TicTacToeHeuristic(
                                    'player_{0}'.format(opponent_id)) if opponent == 'heuristic' else RandomLegal(),
                                'player_{0}'.format(player_number): agent
                            }

                            result = run_pettingzoo_env(env, agents, n_games_train=500, n_games_test=50,
                                                        learning_type=learning_type)

                            new_row = {
                                'learning_type': learning_type.name,
                                'use_model_only': 'Yes' if use_model_only else 'No',
                                'enable_action_blocking': 'Yes' if enable_action_blocking else 'No',
                                'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                'opponent': opponent,
                                'player_number': player_number
                            }

                            for key in result:
                                new_row.update({key: result[key]})

                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, float_format='%.3f', index=False)


def run_random_forest_heuristics(env, env_name, heuristic_func, min_penalty=0, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_rf.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'enable_action_blocking', 'opponent', 'player_number',
                   'use_preloaded_memory', 'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for learning_type in [LearningType.OFFLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for enable_action_blocking in list({False, min_penalty > 0}):
                for use_preloaded_memory in [False, True]:
                    for opponent in ['heuristic', 'random_legal']:
                        for player_number in [1, 2]:
                            opponent_id = (2 - player_number) + 1
                            pre_loaded_memory = develop_memory_from_pettingzoo_env(env, max_time_steps,
                                                                                   'player_{0}'.format(player_number))

                            agent = HeuristicWithRF(env.observation_space.shape, heuristic_func, use_model_only,
                                                    env.action_space,
                                                    enable_action_blocking, min_penalty,
                                                    pre_loaded_memory if use_preloaded_memory else None,
                                                    None, None, max_time_steps, 'random_forest', **args)

                            agents = {
                                'player_{0}'.format(opponent_id): TicTacToeHeuristic(
                                    'player_{0}'.format(opponent_id)) if opponent == 'heuristic' else RandomLegal(),
                                'player_{0}'.format(player_number): agent
                            }

                            result = run_pettingzoo_env(env, agents, n_games_train=500, n_games_test=50,
                                                        learning_type=learning_type)

                            new_row = {
                                'learning_type': learning_type.name,
                                'use_model_only': 'Yes' if use_model_only else 'No',
                                'enable_action_blocking': 'Yes' if enable_action_blocking else 'No',
                                'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                'opponent': opponent,
                                'player_number': player_number
                            }

                            for key in result:
                                new_row.update({key: result[key]})

                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, float_format='%.3f', index=False)


def run_td_epsilon_greedy_heuristics(env, env_name, heuristic_func, penalty, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_td_epsilon_greedy.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size',
                   'optimizer', 'learning_rate', 'goal_focused', 'is_double', 'algorithm_type', 'enable_decay',
                   'epsilon_start', 'add_conservative_loss', 'alpha', 'enable_action_blocker',
                   'action_blocker_model_type', 'use_preloaded_memory', 'use_mse',
                   'opponent', 'player_number',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}
    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for is_double in [False, True]:
                for algorithm_type in TDAlgorithmType.all():
                    for enable_action_blocker in list({False, penalty > 0}):
                        action_blocker_model_types = ['decision_tree', 'random_forest'] if enable_action_blocker else [
                            None]
                        for action_blocker_model_type in action_blocker_model_types:
                            for batch_size in [32, 64, 128]:
                                for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                                    for learning_rate in [0.001, 0.0001]:
                                        for hidden_layer_size in [64, 128, 256, 512]:
                                            for add_conservative_loss in [False, True]:
                                                for alpha in [0.001]:
                                                    for enable_decay in [False, True]:
                                                        epsilons = [1.0] if enable_decay \
                                                            else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                                                        for epsilon in epsilons:
                                                            for use_preloaded_memory in [False, True]:
                                                                for use_mse in [False, True]:
                                                                    for opponent in ['heuristic', 'random_legal']:
                                                                        for player_number in [1, 2]:
                                                                            opponent_id = (2 - player_number) + 1
                                                                            pre_loaded_memory = develop_memory_from_pettingzoo_env(
                                                                                env, max_time_steps,
                                                                                'player_{0}'.format(player_number))
                                                                            policy_args.update(
                                                                                {'eps_start': epsilon,
                                                                                 'enable_decay': enable_decay})
                                                                            network_optimizer_args = {
                                                                                'learning_rate': learning_rate
                                                                            }
                                                                            network_args = {
                                                                                'fc_dims': hidden_layer_size
                                                                            }
                                                                            agent = HeuristicWithTD(
                                                                                input_dims=env.observation_space.shape,
                                                                                action_space=env.action_space,
                                                                                gamma=0.99,
                                                                                mem_size=1000000,
                                                                                batch_size=batch_size,
                                                                                network_args=network_args,
                                                                                optimizer_type=optimizer_type,
                                                                                replace=1000,
                                                                                optimizer_args=network_optimizer_args,
                                                                                enable_action_blocking=enable_action_blocker,
                                                                                min_penalty=penalty,
                                                                                is_double=is_double,
                                                                                algorithm_type=algorithm_type,
                                                                                policy_type=PolicyType.EPSILON_GREEDY,
                                                                                policy_args=policy_args,
                                                                                use_model_only=use_model_only,
                                                                                learning_type=learning_type,
                                                                                heuristic_func=heuristic_func,
                                                                                add_conservative_loss=add_conservative_loss,
                                                                                alpha=alpha,
                                                                                action_blocker_model_type=action_blocker_model_type,
                                                                                action_blocker_timesteps=max_time_steps,
                                                                                pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                                                use_mse=use_mse,
                                                                                **args)

                                                                            agents = {
                                                                                'player_{0}'.format(
                                                                                    opponent_id): TicTacToeHeuristic(
                                                                                    'player_{0}'.format(
                                                                                        opponent_id)) if opponent == 'heuristic' else RandomLegal(),
                                                                                'player_{0}'.format(
                                                                                    player_number): agent
                                                                            }

                                                                            result = run_pettingzoo_env(env, agents,
                                                                                                        n_games_train=500,
                                                                                                        n_games_test=50,
                                                                                                        learning_type=learning_type)
                                                                            new_row = {
                                                                                'batch_size': batch_size,
                                                                                'hidden_layer_size': hidden_layer_size,
                                                                                'algorithm_type': algorithm_type,
                                                                                'optimizer': optimizer_type.name.lower(),
                                                                                'learning_rate': learning_rate,
                                                                                'opponent': opponent,
                                                                                'player_number': player_number,
                                                                                'is_double': 'Yes' if is_double else 'No',
                                                                                'enable_decay': 'Yes' if enable_decay else 'No',
                                                                                'epsilon': epsilon,
                                                                                'learning_type': learning_type.name,
                                                                                'use_model_only': 'Yes' if use_model_only else 'No',
                                                                                'add_conservative_loss': 'Yes' if add_conservative_loss else 'No',
                                                                                'alpha': alpha if add_conservative_loss else 0,
                                                                                'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                                                                'action_blocker_model_type': action_blocker_model_type if enable_action_blocker else 'N/A',
                                                                                'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                                                                'use_mse': 'Yes' if use_mse else 'No'
                                                                            }
                                                                            for key in result:
                                                                                new_row.update({key: result[key]})

                                                                            results = results.append(new_row,
                                                                                                     ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td_softmax_heuristics(env, env_name, heuristic_func, penalty, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_td_softmax.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size',
                   'optimizer', 'learning_rate', 'goal_focused', 'is_double', 'algorithm_type', 'tau',
                   'add_conservative_loss', 'alpha', 'enable_action_blocker', 'action_blocker_model_type',
                   'opponent', 'player_number',
                   'use_preloaded_memory', 'use_mse', 'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}
    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for is_double in [False, True]:
                for algorithm_type in TDAlgorithmType.all():
                    for enable_action_blocker in list({False, penalty > 0}):
                        action_blocker_model_types = ['decision_tree', 'random_forest'] if enable_action_blocker else [
                            None]
                        for action_blocker_model_type in action_blocker_model_types:
                            for batch_size in [32, 64, 128]:
                                for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                                    for learning_rate in [0.001, 0.0001]:
                                        for hidden_layer_size in [64, 128, 256, 512]:
                                            for add_conservative_loss in [False, True]:
                                                for alpha in [0.001]:
                                                    for tau in [0.0001, 0.001, 0.1, 1.0, 10.0]:
                                                        for use_preloaded_memory in [False, True]:
                                                            for use_mse in [False, True]:
                                                                for opponent in ['heuristic', 'random_legal']:
                                                                    for player_number in [1, 2]:
                                                                        opponent_id = (2 - player_number) + 1
                                                                        pre_loaded_memory = develop_memory_from_pettingzoo_env(
                                                                            env, max_time_steps,
                                                                            'player_{0}'.format(player_number))
                                                                        policy_args.update({'tau': tau})
                                                                        network_optimizer_args = {
                                                                            'learning_rate': learning_rate
                                                                        }
                                                                        network_args = {
                                                                            'fc_dims': hidden_layer_size
                                                                        }
                                                                        agent = HeuristicWithTD(
                                                                            input_dims=env.observation_space.shape,
                                                                            action_space=env.action_space,
                                                                            gamma=0.99,
                                                                            mem_size=1000000,
                                                                            batch_size=batch_size,
                                                                            network_args=network_args,
                                                                            optimizer_type=optimizer_type,
                                                                            replace=1000,
                                                                            optimizer_args=network_optimizer_args,
                                                                            enable_action_blocking=enable_action_blocker,
                                                                            min_penalty=penalty,
                                                                            is_double=is_double,
                                                                            algorithm_type=algorithm_type,
                                                                            policy_type=PolicyType.SOFTMAX,
                                                                            policy_args=policy_args,
                                                                            heuristic_func=heuristic_func,
                                                                            learning_type=learning_type,
                                                                            add_conservative_loss=add_conservative_loss,
                                                                            alpha=alpha,
                                                                            use_model_only=use_model_only,
                                                                            action_blocker_timesteps=max_time_steps,
                                                                            action_blocker_model_type=action_blocker_model_type,
                                                                            pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                                            use_mse=use_mse,
                                                                            **args)

                                                                        agents = {
                                                                            'player_{0}'.format(
                                                                                opponent_id): TicTacToeHeuristic(
                                                                                'player_{0}'.format(
                                                                                    opponent_id)) if opponent == 'heuristic' else RandomLegal(),
                                                                            'player_{0}'.format(player_number): agent
                                                                        }

                                                                        result = run_pettingzoo_env(env, agents,
                                                                                                    n_games_train=500,
                                                                                                    n_games_test=50,
                                                                                                    learning_type=learning_type)

                                                                        new_row = {
                                                                            'batch_size': batch_size,
                                                                            'hidden_layer_size': hidden_layer_size,
                                                                            'algorithm_type': algorithm_type,
                                                                            'optimizer': optimizer_type.name.lower(),
                                                                            'learning_rate': learning_rate,
                                                                            'opponent': opponent,
                                                                            'player_number': player_number,
                                                                            'is_double': 'Yes' if is_double else 'No',
                                                                            'tau': tau,
                                                                            'use_model_only': 'Yes' if use_model_only else 'No',
                                                                            'learning_type': learning_type.name,
                                                                            'add_conservative_loss': 'Yes' if add_conservative_loss else 'No',
                                                                            'alpha': alpha if add_conservative_loss else 0,
                                                                            'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                                                            'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                                                            'action_blocker_model_type': action_blocker_model_type if enable_action_blocker else 'N/A',
                                                                            'use_mse': 'Yes' if use_mse else 'No'
                                                                        }
                                                                        for key in result:
                                                                            new_row.update({key: result[key]})

                                                                        results = results.append(new_row,
                                                                                                 ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td_ucb_heuristics(env, env_name, heuristic_func, penalty, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_td_ucb.csv'.format(env_name))

    policy_args = {'confidence_factor': 2}

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size',
                   'optimizer', 'learning_rate', 'goal_focused',
                   'is_double', 'algorithm_type', 'add_conservative_loss', 'alpha',
                   'enable_action_blocker', 'action_blocker_model_type', 'use_preloaded_memory', 'use_mse',
                   'opponent', 'player_number',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for is_double in [False, True]:
                for algorithm_type in TDAlgorithmType.all():
                    for enable_action_blocker in list({False, penalty > 0}):
                        action_blocker_model_types = ['decision_tree', 'random_forest'] if enable_action_blocker else [
                            None]
                        for action_blocker_model_type in action_blocker_model_types:
                            for batch_size in [32, 64, 128]:
                                for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                                    for learning_rate in [0.001, 0.0001]:
                                        for hidden_layer_size in [64, 128, 256, 512]:
                                            for add_conservative_loss in [False, True]:
                                                for alpha in [0.001]:
                                                    for use_preloaded_memory in [False, True]:
                                                        for use_mse in [False, True]:
                                                            for opponent in ['heuristic', 'random_legal']:
                                                                for player_number in [1, 2]:
                                                                    opponent_id = (2 - player_number) + 1
                                                                    pre_loaded_memory = develop_memory_from_pettingzoo_env(
                                                                        env, max_time_steps,
                                                                        'player_{0}'.format(player_number))
                                                                    network_optimizer_args = {
                                                                        'learning_rate': learning_rate
                                                                    }
                                                                    network_args = {
                                                                        'fc_dims': hidden_layer_size
                                                                    }
                                                                    agent = HeuristicWithTD(
                                                                        input_dims=env.observation_space.shape,
                                                                        action_space=env.action_space,
                                                                        gamma=0.99,
                                                                        mem_size=1000000,
                                                                        batch_size=batch_size,
                                                                        network_args=network_args,
                                                                        optimizer_type=optimizer_type,
                                                                        replace=1000,
                                                                        optimizer_args=network_optimizer_args,
                                                                        enable_action_blocking=enable_action_blocker,
                                                                        min_penalty=penalty,
                                                                        is_double=is_double,
                                                                        algorithm_type=algorithm_type,
                                                                        policy_type=PolicyType.UCB,
                                                                        policy_args=policy_args,
                                                                        learning_type=learning_type,
                                                                        use_model_only=use_model_only,
                                                                        heuristic_func=heuristic_func,
                                                                        add_conservative_loss=add_conservative_loss,
                                                                        action_blocker_model_type=action_blocker_model_type,
                                                                        action_blocker_timesteps=max_time_steps,
                                                                        pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                                        use_mse=use_mse,
                                                                        alpha=alpha, **args)

                                                                    agents = {
                                                                        'player_{0}'.format(
                                                                            opponent_id): TicTacToeHeuristic(
                                                                            'player_{0}'.format(
                                                                                opponent_id)) if opponent == 'heuristic' else RandomLegal(),
                                                                        'player_{0}'.format(player_number): agent
                                                                    }

                                                                    result = run_pettingzoo_env(env, agents,
                                                                                                n_games_train=500,
                                                                                                n_games_test=50,
                                                                                                learning_type=learning_type)

                                                                    new_row = {
                                                                        'batch_size': batch_size,
                                                                        'hidden_layer_size': hidden_layer_size,
                                                                        'algorithm_type': algorithm_type,
                                                                        'optimizer': optimizer_type.name.lower(),
                                                                        'learning_rate': learning_rate,
                                                                        'opponent': opponent,
                                                                        'player_number': player_number,
                                                                        'is_double': 'Yes' if is_double else 'No',
                                                                        'learning_type': learning_type.name,
                                                                        'use_model_only': 'Yes' if use_model_only else 'No',
                                                                        'add_conservative_loss': 'Yes' if add_conservative_loss else 'No',
                                                                        'alpha': alpha if add_conservative_loss else 0,
                                                                        'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                                                        'action_blocker_model_type': action_blocker_model_type if enable_action_blocker else 'N/A',
                                                                        'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                                                        'use_mse': 'Yes' if use_mse else 'No'
                                                                    }
                                                                    for key in result:
                                                                        new_row.update({key: result[key]})

                                                                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td_thompson_sampling_heuristics(env, env_name, heuristic_func, penalty, **args):
    assert penalty > 0
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_td_thompson_sampling.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size',
                   'optimizer', 'learning_rate', 'goal_focused',
                   'is_double', 'algorithm_type', 'add_conservative_loss', 'alpha',
                   'enable_action_blocker', 'action_blocker_model_type', 'use_preloaded_memory', 'use_mse',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}
    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for is_double in [False, True]:
                for algorithm_type in TDAlgorithmType.all():
                    for enable_action_blocker in list({False, penalty > 0}):
                        action_blocker_model_types = ['decision_tree', 'random_forest'] if enable_action_blocker else [
                            None]
                        for action_blocker_model_type in action_blocker_model_types:
                            for batch_size in [32, 64, 128]:
                                for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                                    for learning_rate in [0.001, 0.0001]:
                                        for hidden_layer_size in [64, 128, 256, 512]:
                                            for add_conservative_loss in [False, True]:
                                                for alpha in [0.001]:
                                                    for use_preloaded_memory in [False, True]:
                                                        for use_mse in [False, True]:
                                                            for opponent in ['heuristic', 'random_legal']:
                                                                for player_number in [1, 2]:
                                                                    opponent_id = (2 - player_number) + 1
                                                                    pre_loaded_memory = develop_memory_from_pettingzoo_env(
                                                                        env, max_time_steps,
                                                                        'player_{0}'.format(player_number))
                                                                    network_optimizer_args = {
                                                                        'learning_rate': learning_rate
                                                                    }
                                                                    network_args = {
                                                                        'fc_dims': hidden_layer_size
                                                                    }
                                                                    agent = HeuristicWithTD(
                                                                        input_dims=env.observation_space.shape,
                                                                        action_space=env.action_space,
                                                                        gamma=0.99,
                                                                        mem_size=1000000,
                                                                        batch_size=batch_size,
                                                                        network_args=network_args,
                                                                        optimizer_type=optimizer_type,
                                                                        replace=1000,
                                                                        optimizer_args=network_optimizer_args,
                                                                        enable_action_blocking=enable_action_blocker,
                                                                        min_penalty=penalty,
                                                                        is_double=is_double,
                                                                        algorithm_type=algorithm_type,
                                                                        policy_type=PolicyType.THOMPSON_SAMPLING,
                                                                        policy_args=policy_args,
                                                                        learning_type=learning_type,
                                                                        use_model_only=use_model_only,
                                                                        heuristic_func=heuristic_func,
                                                                        add_conservative_loss=add_conservative_loss,
                                                                        action_blocker_model_type=action_blocker_model_type,
                                                                        action_blocker_timesteps=max_time_steps,
                                                                        pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                                        alpha=alpha,
                                                                        use_mse=use_mse,
                                                                        **args)

                                                                    agents = {
                                                                        'player_{0}'.format(
                                                                            opponent_id): TicTacToeHeuristic(
                                                                            'player_{0}'.format(
                                                                                opponent_id)) if opponent == 'heuristic' else RandomLegal(),
                                                                        'player_{0}'.format(player_number): agent
                                                                    }

                                                                    result = run_pettingzoo_env(env, agents,
                                                                                                n_games_train=500,
                                                                                                n_games_test=50,
                                                                                                learning_type=learning_type)

                                                                    new_row = {
                                                                        'batch_size': batch_size,
                                                                        'hidden_layer_size': hidden_layer_size,
                                                                        'algorithm_type': algorithm_type,
                                                                        'optimizer': optimizer_type.name.lower(),
                                                                        'learning_rate': learning_rate,
                                                                        'opponent': opponent,
                                                                        'player_number': player_number,
                                                                        'is_double': 'Yes' if is_double else 'No',
                                                                        'learning_type': learning_type.name,
                                                                        'use_model_only': 'Yes' if use_model_only else 'No',
                                                                        'add_conservative_loss': 'Yes' if add_conservative_loss else 'No',
                                                                        'alpha': alpha if add_conservative_loss else 0,
                                                                        'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                                                        'action_blocker_model_type': action_blocker_model_type if enable_action_blocker else 'N/A',
                                                                        'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                                                        'use_mse': 'Yes' if use_mse else 'No'
                                                                    }
                                                                    for key in result:
                                                                        new_row.update({key: result[key]})

                                                                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_epsilon_greedy_heuristics(env, env_name, heuristic_func, penalty, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_dueling_td_epsilon_greedy.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size',
                   'optimizer', 'learning_rate', 'goal_focused',
                   'is_double', 'algorithm_type', 'enable_decay', 'epsilon_start',
                   'add_conservative_loss', 'alpha', 'enable_action_blocker', 'action_blocker_model_type',
                   'opponent', 'player_number',
                   'use_preloaded_memory', 'use_mse', 'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}
    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for is_double in [False, True]:
                for algorithm_type in TDAlgorithmType.all():
                    for enable_action_blocker in list({False, penalty > 0}):
                        action_blocker_model_types = ['decision_tree', 'random_forest'] if enable_action_blocker else [
                            None]
                        for action_blocker_model_type in action_blocker_model_types:
                            for batch_size in [32, 64, 128]:
                                for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                                    for learning_rate in [0.001, 0.0001]:
                                        for hidden_layer_size in [64, 128, 256, 512]:
                                            for add_conservative_loss in [False, True]:
                                                for alpha in [0.001]:
                                                    for enable_decay in [False, True]:
                                                        epsilons = [1.0] if enable_decay \
                                                            else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                                                        for epsilon in epsilons:
                                                            for use_preloaded_memory in [False, True]:
                                                                for use_mse in [False, True]:
                                                                    for opponent in ['heuristic', 'random_legal']:
                                                                        for player_number in [1, 2]:
                                                                            opponent_id = (2 - player_number) + 1
                                                                            pre_loaded_memory = develop_memory_from_pettingzoo_env(
                                                                                env, max_time_steps,
                                                                                'player_{0}'.format(player_number))

                                                                            policy_args.update(
                                                                                {'eps_start': epsilon,
                                                                                 'enable_decay': enable_decay})
                                                                            network_optimizer_args = {
                                                                                'learning_rate': learning_rate
                                                                            }
                                                                            network_args = {
                                                                                'fc_dims': hidden_layer_size
                                                                            }
                                                                            agent = HeuristicWithDuelingTD(
                                                                                input_dims=env.observation_space.shape,
                                                                                action_space=env.action_space,
                                                                                gamma=0.99,
                                                                                mem_size=1000000,
                                                                                batch_size=batch_size,
                                                                                network_args=network_args,
                                                                                optimizer_type=optimizer_type,
                                                                                replace=1000,
                                                                                optimizer_args=network_optimizer_args,
                                                                                enable_action_blocking=enable_action_blocker,
                                                                                min_penalty=penalty,
                                                                                is_double=is_double,
                                                                                algorithm_type=algorithm_type,
                                                                                policy_type=PolicyType.EPSILON_GREEDY,
                                                                                policy_args=policy_args,
                                                                                use_model_only=use_model_only,
                                                                                learning_type=learning_type,
                                                                                heuristic_func=heuristic_func,
                                                                                add_conservative_loss=add_conservative_loss,
                                                                                action_blocker_model_type=action_blocker_model_type,
                                                                                action_blocker_timesteps=max_time_steps,
                                                                                pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                                                alpha=alpha,
                                                                                use_mse=use_mse,
                                                                                **args)

                                                                            agents = {
                                                                                'player_{0}'.format(
                                                                                    opponent_id): TicTacToeHeuristic(
                                                                                    'player_{0}'.format(
                                                                                        opponent_id)) if opponent == 'heuristic' else RandomLegal(),
                                                                                'player_{0}'.format(
                                                                                    player_number): agent
                                                                            }

                                                                            result = run_pettingzoo_env(env, agents,
                                                                                                        n_games_train=500,
                                                                                                        n_games_test=50,
                                                                                                        learning_type=learning_type)
                                                                            new_row = {
                                                                                'batch_size': batch_size,
                                                                                'hidden_layer_size': hidden_layer_size,
                                                                                'algorithm_type': algorithm_type,
                                                                                'optimizer': optimizer_type.name.lower(),
                                                                                'learning_rate': learning_rate,
                                                                                'opponent': opponent,
                                                                                'player_number': player_number,
                                                                                'is_double': 'Yes' if is_double else 'No',
                                                                                'enable_decay': 'Yes' if enable_decay else 'No',
                                                                                'epsilon': epsilon,
                                                                                'learning_type': learning_type.name,
                                                                                'use_model_only': 'Yes' if use_model_only else 'No',
                                                                                'add_conservative_loss': 'Yes' if add_conservative_loss else 'No',
                                                                                'alpha': alpha if add_conservative_loss else 0,
                                                                                'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                                                                'action_blocker_model_type': action_blocker_model_type if enable_action_blocker else 'N/A',
                                                                                'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                                                                'use_mse': 'Yes' if use_mse else 'No'
                                                                            }
                                                                            for key in result:
                                                                                new_row.update({key: result[key]})

                                                                            results = results.append(new_row,
                                                                                                     ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_softmax_heuristics(env, env_name, heuristic_func, penalty, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_dueling_td_softmax.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size',
                   'optimizer', 'learning_rate', 'goal_focused', 'is_double', 'algorithm_type', 'tau',
                   'add_conservative_loss', 'alpha', 'enable_action_blocker', 'action_blocker_model_type',
                   'opponent', 'player_number',
                   'use_preloaded_memory', 'use_mse', 'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}
    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for is_double in [False, True]:
                for algorithm_type in TDAlgorithmType.all():
                    for enable_action_blocker in list({False, penalty > 0}):
                        action_blocker_model_types = ['decision_tree', 'random_forest'] if enable_action_blocker else [
                            None]
                        for action_blocker_model_type in action_blocker_model_types:
                            for batch_size in [32, 64, 128]:
                                for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                                    for learning_rate in [0.001, 0.0001]:
                                        for hidden_layer_size in [64, 128, 256, 512]:
                                            for add_conservative_loss in [False, True]:
                                                for alpha in [0.001]:
                                                    for tau in [0.0001, 0.001, 0.1, 1.0, 10.0]:
                                                        for use_preloaded_memory in [False, True]:
                                                            for use_mse in [False, True]:
                                                                for opponent in ['heuristic', 'random_legal']:
                                                                    for player_number in [1, 2]:
                                                                        opponent_id = (2 - player_number) + 1
                                                                        pre_loaded_memory = develop_memory_from_pettingzoo_env(
                                                                            env, max_time_steps,
                                                                            'player_{0}'.format(player_number))
                                                                        policy_args.update({'tau': tau})
                                                                        network_optimizer_args = {
                                                                            'learning_rate': learning_rate
                                                                        }
                                                                        network_args = {
                                                                            'fc_dims': hidden_layer_size
                                                                        }
                                                                        agent = HeuristicWithDuelingTD(
                                                                            input_dims=env.observation_space.shape,
                                                                            action_space=env.action_space,
                                                                            gamma=0.99,
                                                                            mem_size=1000000,
                                                                            batch_size=batch_size,
                                                                            network_args=network_args,
                                                                            optimizer_type=optimizer_type,
                                                                            replace=1000,
                                                                            optimizer_args=network_optimizer_args,
                                                                            enable_action_blocking=enable_action_blocker,
                                                                            min_penalty=penalty,
                                                                            is_double=is_double,
                                                                            algorithm_type=algorithm_type,
                                                                            policy_type=PolicyType.SOFTMAX,
                                                                            policy_args=policy_args,
                                                                            heuristic_func=heuristic_func,
                                                                            learning_type=learning_type,
                                                                            add_conservative_loss=add_conservative_loss,
                                                                            alpha=alpha,
                                                                            action_blocker_model_type=action_blocker_model_type,
                                                                            action_blocker_timesteps=max_time_steps,
                                                                            pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                                            use_model_only=use_model_only,
                                                                            use_mse=use_mse,
                                                                            **args)

                                                                        agents = {
                                                                            'player_{0}'.format(
                                                                                opponent_id): TicTacToeHeuristic(
                                                                                'player_{0}'.format(
                                                                                    opponent_id)) if opponent == 'heuristic' else RandomLegal(),
                                                                            'player_{0}'.format(player_number): agent
                                                                        }

                                                                        result = run_pettingzoo_env(env, agents,
                                                                                                    n_games_train=500,
                                                                                                    n_games_test=50,
                                                                                                    learning_type=learning_type)

                                                                        new_row = {
                                                                            'batch_size': batch_size,
                                                                            'hidden_layer_size': hidden_layer_size,
                                                                            'algorithm_type': algorithm_type,
                                                                            'optimizer': optimizer_type.name.lower(),
                                                                            'learning_rate': learning_rate,
                                                                            'opponent': opponent,
                                                                            'player_number': player_number,
                                                                            'is_double': 'Yes' if is_double else 'No',
                                                                            'tau': tau,
                                                                            'use_model_only': 'Yes' if use_model_only else 'No',
                                                                            'learning_type': learning_type.name,
                                                                            'add_conservative_loss': 'Yes' if add_conservative_loss else 'No',
                                                                            'alpha': alpha if add_conservative_loss else 0,
                                                                            'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                                                            'action_blocker_model_type': action_blocker_model_type if enable_action_blocker else 'N/A',
                                                                            'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                                                            'use_mse': 'Yes' if use_mse else 'No'
                                                                        }
                                                                        for key in result:
                                                                            new_row.update({key: result[key]})

                                                                        results = results.append(new_row,
                                                                                                 ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_ucb_heuristics(env, env_name, heuristic_func, penalty, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_dueling_td_ucb.csv'.format(env_name))

    policy_args = {'confidence_factor': 2}

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size',
                   'optimizer', 'learning_rate', 'goal_focused',
                   'is_double', 'algorithm_type', 'add_conservative_loss', 'alpha',
                   'enable_action_blocker', 'action_blocker_model_type', 'use_preloaded_memory', 'use_mse',
                   'opponent', 'player_number',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for is_double in [False, True]:
                for algorithm_type in TDAlgorithmType.all():
                    for enable_action_blocker in list({False, penalty > 0}):
                        action_blocker_model_types = ['decision_tree', 'random_forest'] if enable_action_blocker else [
                            None]
                        for action_blocker_model_type in action_blocker_model_types:
                            for batch_size in [32, 64, 128]:
                                for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                                    for learning_rate in [0.001, 0.0001]:
                                        for hidden_layer_size in [64, 128, 256, 512]:
                                            for add_conservative_loss in [False, True]:
                                                for alpha in [0.001]:
                                                    for use_preloaded_memory in [False, True]:
                                                        for use_mse in [False, True]:
                                                            for opponent in ['heuristic', 'random_legal']:
                                                                for player_number in [1, 2]:
                                                                    opponent_id = (2 - player_number) + 1
                                                                    pre_loaded_memory = develop_memory_from_pettingzoo_env(
                                                                        env, max_time_steps,
                                                                        'player_{0}'.format(player_number))
                                                                    network_optimizer_args = {
                                                                        'learning_rate': learning_rate
                                                                    }
                                                                    network_args = {
                                                                        'fc_dims': hidden_layer_size
                                                                    }
                                                                    agent = HeuristicWithDuelingTD(
                                                                        input_dims=env.observation_space.shape,
                                                                        action_space=env.action_space,
                                                                        gamma=0.99,
                                                                        mem_size=1000000,
                                                                        batch_size=batch_size,
                                                                        network_args=network_args,
                                                                        optimizer_type=optimizer_type,
                                                                        replace=1000,
                                                                        optimizer_args=network_optimizer_args,
                                                                        enable_action_blocking=enable_action_blocker,
                                                                        min_penalty=penalty,
                                                                        is_double=is_double,
                                                                        algorithm_type=algorithm_type,
                                                                        policy_type=PolicyType.UCB,
                                                                        policy_args=policy_args,
                                                                        learning_type=learning_type,
                                                                        use_model_only=use_model_only,
                                                                        heuristic_func=heuristic_func,
                                                                        add_conservative_loss=add_conservative_loss,
                                                                        action_blocker_model_type=action_blocker_model_type,
                                                                        action_blocker_timesteps=max_time_steps,
                                                                        pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                                        alpha=alpha, use_mse=use_mse, **args)

                                                                    agents = {
                                                                        'player_{0}'.format(
                                                                            opponent_id): TicTacToeHeuristic(
                                                                            'player_{0}'.format(
                                                                                opponent_id)) if opponent == 'heuristic' else RandomLegal(),
                                                                        'player_{0}'.format(player_number): agent
                                                                    }

                                                                    result = run_pettingzoo_env(env, agents,
                                                                                                n_games_train=500,
                                                                                                n_games_test=50,
                                                                                                learning_type=learning_type)

                                                                    new_row = {
                                                                        'batch_size': batch_size,
                                                                        'hidden_layer_size': hidden_layer_size,
                                                                        'algorithm_type': algorithm_type,
                                                                        'optimizer': optimizer_type.name.lower(),
                                                                        'learning_rate': learning_rate,
                                                                        'opponent': opponent,
                                                                        'player_number': player_number,
                                                                        'is_double': 'Yes' if is_double else 'No',
                                                                        'learning_type': learning_type.name,
                                                                        'use_model_only': 'Yes' if use_model_only else 'No',
                                                                        'add_conservative_loss': 'Yes' if add_conservative_loss else 'No',
                                                                        'alpha': alpha if add_conservative_loss else 0,
                                                                        'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                                                        'action_blocker_model_type': action_blocker_model_type if enable_action_blocker else 'None',
                                                                        'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                                                        'use_mse': 'Yes' if use_mse else 'No'
                                                                    }
                                                                    for key in result:
                                                                        new_row.update({key: result[key]})

                                                                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_thompson_sampling_heuristics(env, env_name, heuristic_func, penalty, **args):
    assert penalty > 0
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_dueling_td_thompson_sampling.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size',
                   'optimizer', 'learning_rate', 'goal_focused',
                   'is_double', 'algorithm_type', 'add_conservative_loss', 'alpha',
                   'enable_action_blocker', 'action_blocker_model_type', 'use_preloaded_memory', 'use_mse',
                   'opponent', 'player_number',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}
    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for is_double in [False, True]:
                for algorithm_type in TDAlgorithmType.all():
                    for enable_action_blocker in list({False, penalty > 0}):
                        action_blocker_model_types = ['decision_tree', 'random_forest'] if enable_action_blocker else [
                            None]
                        for action_blocker_model_type in action_blocker_model_types:
                            for batch_size in [32, 64, 128]:
                                for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                                    for learning_rate in [0.001, 0.0001]:
                                        for hidden_layer_size in [64, 128, 256, 512]:
                                            for add_conservative_loss in [False, True]:
                                                for alpha in [0.001]:
                                                    for use_preloaded_memory in [False, True]:
                                                        for use_mse in [False, True]:
                                                            for opponent in ['heuristic', 'random_legal']:
                                                                for player_number in [1, 2]:
                                                                    opponent_id = (2 - player_number) + 1
                                                                    pre_loaded_memory = develop_memory_from_pettingzoo_env(
                                                                        env, max_time_steps,
                                                                        'player_{0}'.format(player_number))
                                                                    network_optimizer_args = {
                                                                        'learning_rate': learning_rate
                                                                    }
                                                                    network_args = {
                                                                        'fc_dims': hidden_layer_size
                                                                    }
                                                                    agent = HeuristicWithDuelingTD(
                                                                        input_dims=env.observation_space.shape,
                                                                        action_space=env.action_space,
                                                                        gamma=0.99,
                                                                        mem_size=1000000,
                                                                        batch_size=batch_size,
                                                                        network_args=network_args,
                                                                        optimizer_type=optimizer_type,
                                                                        replace=1000,
                                                                        optimizer_args=network_optimizer_args,
                                                                        enable_action_blocking=enable_action_blocker,
                                                                        min_penalty=penalty,
                                                                        is_double=is_double,
                                                                        algorithm_type=algorithm_type,
                                                                        policy_type=PolicyType.THOMPSON_SAMPLING,
                                                                        policy_args=policy_args,
                                                                        learning_type=learning_type,
                                                                        use_model_only=use_model_only,
                                                                        heuristic_func=heuristic_func,
                                                                        add_conservative_loss=add_conservative_loss,
                                                                        action_blocker_model_type=action_blocker_model_type,
                                                                        action_blocker_timesteps=max_time_steps,
                                                                        pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                                        alpha=alpha, use_mse=use_mse, **args)

                                                                    agents = {
                                                                        'player_{0}'.format(
                                                                            opponent_id): TicTacToeHeuristic(
                                                                            'player_{0}'.format(
                                                                                opponent_id)) if opponent == 'heuristic' else RandomLegal(),
                                                                        'player_{0}'.format(player_number): agent
                                                                    }

                                                                    result = run_pettingzoo_env(env, agents,
                                                                                                n_games_train=500,
                                                                                                n_games_test=50,
                                                                                                learning_type=learning_type)

                                                                    new_row = {
                                                                        'batch_size': batch_size,
                                                                        'hidden_layer_size': hidden_layer_size,
                                                                        'algorithm_type': algorithm_type,
                                                                        'optimizer': optimizer_type.name.lower(),
                                                                        'learning_rate': learning_rate,
                                                                        'opponent': opponent,
                                                                        'player_number': player_number,
                                                                        'is_double': 'Yes' if is_double else 'No',
                                                                        'learning_type': learning_type.name,
                                                                        'use_model_only': 'Yes' if use_model_only else 'No',
                                                                        'add_conservative_loss': 'Yes' if add_conservative_loss else 'No',
                                                                        'alpha': alpha if add_conservative_loss else 0,
                                                                        'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                                                        'action_blocker_model_type': action_blocker_model_type if enable_action_blocker else 'N/A',
                                                                        'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                                                        'use_mse': 'Yes' if use_mse else 'No'
                                                                    }
                                                                    for key in result:
                                                                        new_row.update({key: result[key]})

                                                                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_hill_climbing_heuristics(env, env_name, penalty, heuristic_func, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_hill_climbing.csv'.format(env_name))
    result_cols = ['use_model_only', 'learning_type', 'enable_action_blocker', 'action_blocker_model_type',
                   'use_preloaded_memory', 'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for use_model_only in [False, True]:
        for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
            for enable_action_blocker in list({False, penalty > 0}):
                action_blocker_model_types = ['decision_tree', 'random_forest'] if enable_action_blocker else [None]
                for action_blocker_model_type in action_blocker_model_types:
                    for use_preloaded_memory in [False, True]:
                        for opponent in ['heuristic', 'random_legal']:
                            for player_number in [1, 2]:
                                opponent_id = (2 - player_number) + 1
                                pre_loaded_memory = develop_memory_from_pettingzoo_env(env, max_time_steps,
                                                                                       'player_{0}'.format(
                                                                                           player_number))
                                agent = HeuristicWithHillClimbing(input_dims=env.observation_space.shape,
                                                                  action_space=env.action_space,
                                                                  enable_action_blocking=enable_action_blocker,
                                                                  action_blocker_model_type=action_blocker_model_type,
                                                                  action_blocker_timesteps=max_time_steps,
                                                                  pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                                  gamma=1.0,
                                                                  heuristic_func=heuristic_func,
                                                                  use_model_only=use_model_only, **args)

                                agents = {
                                    'player_{0}'.format(opponent_id): TicTacToeHeuristic(
                                        'player_{0}'.format(opponent_id)) if opponent == 'heuristic' else RandomLegal(),
                                    'player_{0}'.format(player_number): agent
                                }

                                new_row = {'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                           'use_model_only': 'Yes' if use_model_only else 'No',
                                           'learning_type': learning_type.name,
                                           'use_preloaded_memory': 'Yes' if pre_loaded_memory else 'No',
                                           'action_blocker_model_type': action_blocker_model_type if enable_action_blocker else 'N/A',
                                           'opponent': opponent,
                                           'player_number': player_number
                                           }

                                result = run_pettingzoo_env(env, agents, n_games_train=500, n_games_test=50,
                                                            learning_type=learning_type)

                                for key in result:
                                    new_row.update({key: result[key]})

                                results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_heuristics(env, env_name, heuristic_func, penalty, **args):
    run_decision_tree_heuristics(env, env_name, heuristic_func, penalty, **args)
    run_random_forest_heuristics(env, env_name, heuristic_func, penalty, **args)
    run_td_epsilon_greedy_heuristics(env, env_name, heuristic_func, penalty, **args)
    run_dueling_td_epsilon_greedy_heuristics(env, env_name, heuristic_func, penalty, **args)
    run_td_softmax_heuristics(env, env_name, heuristic_func, penalty, **args)
    run_dueling_td_softmax_heuristics(env, env_name, heuristic_func, penalty, **args)
    run_td_ucb_heuristics(env, env_name, heuristic_func, penalty, **args)
    run_dueling_td_ucb_heuristics(env, env_name, heuristic_func, penalty, **args)
    run_td_thompson_sampling_heuristics(env, env_name, heuristic_func, penalty, **args)
    run_dueling_td_thompson_sampling_heuristics(env, env_name, heuristic_func, penalty, **args)
    run_hill_climbing_heuristics(env, env_name, penalty, heuristic_func, **args)


def tic_tac_toe_heuristic_func(self, observation):
    my_actions_taken = []
    other_actions_taken = []

    state = observation['observation']
    legal_actions = np.where(observation['action_mask'] == 1)[0]

    for player_id in range(2):
        player_grid = state[:, :, player_id]
        where = np.where(player_grid == 1)
        x, y = where
        for i in range(x.shape[0]):
            action_taken = 3 * x[i] + y[i]
            if player_id == 0:
                my_actions_taken.append(action_taken)
            else:
                other_actions_taken.append(action_taken)

    action_to_stop_opponent = self.get_tactical_action(self, legal_actions, other_actions_taken)
    action_to_win_game = self.get_tactical_action(self, legal_actions, my_actions_taken)

    if action_to_stop_opponent:
        return action_to_stop_opponent
    elif action_to_win_game:
        return action_to_win_game
    else:
        if 4 in legal_actions:
            return 4
        elif 0 in legal_actions:
            return 0
        elif 2 in legal_actions:
            return 2
        elif 6 in legal_actions:
            return 6
        elif 8 in legal_actions:
            return 8
        else:
            return np.random.choice(legal_actions)


def get_tactical_action(self, legal_actions, actions_already_taken):
    if len(actions_already_taken) == 0:
        return None

    possible_actions = []
    for i in actions_already_taken:
        for j in actions_already_taken:
            t = (i, j)
            if t in self.actions_can_be_taken:
                possible_actions.append(self.actions_can_be_taken[t])
    if len(possible_actions) == 0:
        return None
    else:
        available_actions = np.intersect1d(np.array(possible_actions), legal_actions)
        return available_actions[0] if available_actions.shape[0] > 0 else None


env = tictactoe_v3.env()
run_all_td_methods(env, 'tic_tact_toe', 1)
run_hill_climbing(env, 'tic_tac_toe', 1)
run_heuristics(env, 'tic_tac_toe', heuristic_func=tic_tac_toe_heuristic_func, penalty=1, actions_can_be_taken={
    (0, 1): 2, (0, 2): 1, (0, 3): 6, (0, 4): 8, (0, 6): 3, (0, 8): 4,
    (1, 0): 2, (1, 2): 0, (1, 4): 7, (1, 7): 4,
    (2, 0): 1, (2, 1): 0, (2, 4): 6, (2, 5): 8, (2, 6): 4, (2, 8): 5,
    (3, 0): 6, (3, 4): 5, (3, 5): 4, (3, 6): 0,
    (4, 0): 8, (4, 1): 7, (4, 2): 6, (4, 5): 3, (4, 6): 2, (4, 7): 1, (4, 8): 0,
    (5, 2): 8, (5, 3): 4, (5, 4): 3, (5, 8): 2,
    (6, 0): 3, (6, 2): 4, (6, 3): 0, (6, 4): 2, (6, 7): 8, (6, 8): 7,
    (7, 1): 4, (7, 4): 1, (7, 6): 8, (7, 8): 6,
    (8, 0): 4, (8, 2): 5, (8, 4): 0, (8, 5): 2, (8, 6): 7, (8, 7): 6
}, get_tactical_action=get_tactical_action)
