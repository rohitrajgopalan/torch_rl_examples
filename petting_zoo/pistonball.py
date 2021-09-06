import os

import numpy as np
import pandas as pd
from pettingzoo.butterfly import pistonball_v4
from torch_rl.ddpg.agent import DDPGAgent
from torch_rl.dueling_td.agent import DuelingTDAgent
from torch_rl.heuristic.heuristic_with_cem import HeuristicWithCEM
from torch_rl.heuristic.heuristic_with_ddpg import HeuristicWithDDPG
from torch_rl.heuristic.heuristic_with_dt import HeuristicWithDT
from torch_rl.heuristic.heuristic_with_dueling_td import HeuristicWithDuelingTD
from torch_rl.heuristic.heuristic_with_td import HeuristicWithTD
from torch_rl.heuristic.heuristic_with_td3 import HeuristicWithTD3
from torch_rl.td.agent import TDAgent
from torch_rl.td3.agent import TD3Agent
from torch_rl.utils.types import NetworkOptimizer, TDAlgorithmType, PolicyType, LearningType

from common.run import run_pettingzoo_env


def run_td_epsilon_greedy(env, env_name):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_td_epsilon_greedy.csv'.format(env_name))

    result_cols = ['batch_size', 'optimizer', 'learning_rate',
                   'assign_priority', 'is_double', 'algorithm_type', 'enable_decay', 'epsilon_start',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}

    network_args = {
        'fc_dim': 512,
        'cnn_dims': [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    }

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for batch_size in [32, 64, 128]:
                for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                    for learning_rate in [0.001, 0.0001]:
                        for enable_decay in [False, True]:
                            epsilons = [1.0] if enable_decay \
                                else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                            for epsilon in epsilons:
                                policy_args.update({'eps_start': epsilon, 'enable_decay': enable_decay})
                                for assign_priority in [False, True]:
                                    network_optimizer_args = {
                                        'learning_rate': learning_rate
                                    }
                                    agents = {}

                                    for agent_id in env.possible_agents:
                                        agents.update({agent_id: TDAgent(
                                            input_dims=env.observation_spaces[agent_id].shape,
                                            action_space=env.action_spaces[agent_id],
                                            gamma=0.99,
                                            mem_size=1000,
                                            batch_size=batch_size,
                                            network_args=network_args,
                                            optimizer_type=optimizer_type,
                                            replace=1000,
                                            optimizer_args=network_optimizer_args,
                                            is_double=is_double,
                                            algorithm_type=algorithm_type,
                                            policy_type=PolicyType.EPSILON_GREEDY,
                                            policy_args=policy_args,
                                            assign_priority=assign_priority)
                                        })

                                    result = run_pettingzoo_env(env, agents, n_games_train=500,
                                                                n_games_test=50)

                                    new_row = {
                                        'batch_size': batch_size,
                                        'algorithm_type': algorithm_type,
                                        'optimizer': optimizer_type.name.lower(),
                                        'learning_rate': learning_rate,
                                        'is_double': 'Yes' if is_double else 'No',
                                        'enable_decay': 'Yes' if enable_decay else 'No',
                                        'epsilon': epsilon,
                                        'assign_priority': 'Yes' if assign_priority else 'No'
                                    }
                                    for key in result:
                                        new_row.update({key: result[key]})

                                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td_softmax(env, env_name):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_td_softmax.csv'.format(env_name))

    result_cols = ['batch_size', 'optimizer', 'learning_rate',
                   'assign_priority', 'is_double', 'algorithm_type', 'tau',
                   'num_time_steps_test', 'avg_score_test']
    results = pd.DataFrame(columns=result_cols)

    policy_args = {}

    network_args = {
        'fc_dim': 512,
        'cnn_dims': [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    }

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for batch_size in [32, 64, 128]:
                for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                    for learning_rate in [0.001, 0.0001]:
                        for tau in [0.0001, 0.001, 0.1, 1.0, 10.0]:
                            policy_args.update({'tau': tau})
                            for assign_priority in [False, True]:
                                network_optimizer_args = {
                                    'learning_rate': learning_rate
                                }
                                agents = {}

                                for agent_id in env.possible_agents:
                                    agents.update({agent_id: TDAgent(
                                        input_dims=env.observation_spaces[agent_id].shape,
                                        action_space=env.action_spaces[agent_id],
                                        gamma=0.99,
                                        mem_size=1000,
                                        batch_size=batch_size,
                                        network_args=network_args,
                                        optimizer_type=optimizer_type,
                                        replace=1000,
                                        optimizer_args=network_optimizer_args,
                                        is_double=is_double,
                                        algorithm_type=algorithm_type,
                                        policy_type=PolicyType.SOFTMAX,
                                        policy_args=policy_args,
                                        assign_priority=assign_priority)})

                                result = run_pettingzoo_env(env, agents, n_games_train=500, n_games_test=50)

                                new_row = {
                                    'batch_size': batch_size,
                                    'algorithm_type': algorithm_type,
                                    'optimizer': optimizer_type.name.lower(),
                                    'learning_rate': learning_rate,
                                    'is_double': 'Yes' if is_double else 'No',
                                    'tau': tau
                                }
                                for key in result:
                                    new_row.update({key: result[key]})

                                results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td_ucb(env, env_name):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results', '{0}_td_ucb.csv'.format(env_name))

    result_cols = ['batch_size', 'optimizer', 'learning_rate',
                   'assign_priority', 'is_double', 'algorithm_type',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {'confidence_factor': 2}

    network_args = {
        'fc_dim': 512,
        'cnn_dims': [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    }

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for batch_size in [32, 64, 128]:
                for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                    for learning_rate in [0.001, 0.0001]:
                        for assign_priority in [False, True]:
                            network_optimizer_args = {
                                'learning_rate': learning_rate
                            }
                            agents = {}

                            for agent_id in env.possible_agents:
                                agents.update({agent_id: TDAgent(
                                    input_dims=env.observation_spaces[agent_id].shape,
                                    action_space=env.action_spaces[agent_id],
                                    gamma=0.99,
                                    mem_size=1000,
                                    batch_size=batch_size,
                                    network_args=network_args,
                                    optimizer_type=optimizer_type,
                                    replace=1000,
                                    optimizer_args=network_optimizer_args,
                                    is_double=is_double,
                                    algorithm_type=algorithm_type,
                                    policy_type=PolicyType.UCB,
                                    policy_args=policy_args,
                                    assign_priority=assign_priority)})

                            result = run_pettingzoo_env(env, agents, n_games_train=500, n_games_test=50)

                            new_row = {
                                'batch_size': batch_size,
                                'algorithm_type': algorithm_type,
                                'optimizer': optimizer_type.name.lower(),
                                'learning_rate': learning_rate,
                                'is_double': 'Yes' if is_double else 'No',
                                'assign_priority': 'Yes' if assign_priority else 'No',
                            }
                            for key in result:
                                new_row.update({key: result[key]})

                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_epsilon_greedy(env, env_name):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_td_epsilon_greedy.csv'.format(env_name))

    result_cols = ['batch_size', 'optimizer', 'learning_rate',
                   'is_double', 'algorithm_type', 'enable_decay',
                   'epsilon_start', 'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}
    network_args = {
        'fc_dim': (1024, 512),
        'cnn_dims': [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    }

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for batch_size in [32, 64, 128]:
                for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                    for learning_rate in [0.001, 0.0001]:
                        for enable_decay in [False, True]:
                            epsilons = [1.0] if enable_decay \
                                else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                            for epsilon in epsilons:
                                policy_args.update({'eps_start': epsilon, 'enable_decay': enable_decay})
                                for assign_priority in [False, True]:
                                    network_optimizer_args = {
                                        'learning_rate': learning_rate
                                    }
                                    agents = {}

                                    for agent_id in env.possible_agents:
                                        agents.update({agent_id: DuelingTDAgent(
                                            input_dims=env.observation_spaces[agent_id].shape,
                                            action_space=env.action_spaces[agent_id],
                                            gamma=0.99,
                                            mem_size=1000,
                                            batch_size=batch_size,
                                            network_args=network_args,
                                            optimizer_type=optimizer_type,
                                            replace=1000,
                                            optimizer_args=network_optimizer_args,
                                            is_double=is_double,
                                            algorithm_type=algorithm_type,
                                            policy_type=PolicyType.EPSILON_GREEDY,
                                            policy_args=policy_args,
                                            assign_priority=assign_priority)})

                                    result = run_pettingzoo_env(env, agents, n_games_train=500, n_games_test=50)

                                    new_row = {
                                        'batch_size': batch_size,
                                        'algorithm_type': algorithm_type,
                                        'optimizer': optimizer_type.name.lower(),
                                        'learning_rate': learning_rate,
                                        'is_double': 'Yes' if is_double else 'No',
                                        'enable_decay': 'Yes' if enable_decay else 'No',
                                        'epsilon': epsilon,
                                        'assign_priority': 'Yes' if assign_priority else 'No'
                                    }
                                    for key in result:
                                        new_row.update({key: result[key]})

                                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_softmax(env, env_name):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_td_softmax.csv'.format(env_name))

    result_cols = ['batch_size', 'optimizer', 'learning_rate',
                   'assign_priority', 'is_double', 'algorithm_type', 'tau',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}
    network_args = {
        'fc_dim': (1024, 512),
        'cnn_dims': [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    }

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for batch_size in [32, 64, 128]:
                for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                    for learning_rate in [0.001, 0.0001]:
                        for tau in [0.0001, 0.001, 0.1, 1.0, 10.0]:
                            policy_args.update({'tau': tau})
                            for assign_priority in [False, True]:
                                network_optimizer_args = {
                                    'learning_rate': learning_rate
                                }
                                agents = {}

                                for agent_id in env.possible_agents:
                                    agents.update({agent_id: DuelingTDAgent(
                                        input_dims=env.observation_spaces[agent_id].shape,
                                        action_space=env.action_spaces[agent_id],
                                        gamma=0.99,
                                        mem_size=1000,
                                        batch_size=batch_size,
                                        network_args=network_args,
                                        optimizer_type=optimizer_type,
                                        replace=1000,
                                        optimizer_args=network_optimizer_args,
                                        is_double=is_double,
                                        algorithm_type=algorithm_type,
                                        policy_type=PolicyType.SOFTMAX,
                                        policy_args=policy_args,
                                        assign_priority=assign_priority)})

                                result = run_pettingzoo_env(env, agents, n_games_train=500, n_games_test=50)

                                new_row = {
                                    'batch_size': batch_size,
                                    'algorithm_type': algorithm_type,
                                    'optimizer': optimizer_type.name.lower(),
                                    'learning_rate': learning_rate,
                                    'is_double': 'Yes' if is_double else 'No',
                                    'tau': tau,
                                    'assign_priority': 'Yes' if assign_priority else 'No'
                                }
                                for key in result:
                                    new_row.update({key: result[key]})

                                results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_ucb(env, env_name):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_td_ucb.csv'.format(env_name))
    result_cols = ['batch_size', 'optimizer', 'learning_rate',
                   'assign_priority', 'is_double', 'algorithm_type',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {'confidence_factor': 2}
    network_args = {
        'fc_dim': (1024, 512),
        'cnn_dims': [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    }

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for batch_size in [32, 64, 128]:
                for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                    for learning_rate in [0.001, 0.0001]:
                        for assign_priority in [False, True]:
                            network_optimizer_args = {
                                'learning_rate': learning_rate
                            }
                            agents = {}

                            for agent_id in env.possible_agents:
                                agents.update({agent_id: DuelingTDAgent(
                                    input_dims=env.observation_spaces[agent_id].shape,
                                    action_space=env.action_spaces[agent_id],
                                    gamma=0.99,
                                    mem_size=1000,
                                    batch_size=batch_size,
                                    network_args=network_args,
                                    optimizer_type=optimizer_type,
                                    replace=1000,
                                    optimizer_args=network_optimizer_args,
                                    is_double=is_double,
                                    algorithm_type=algorithm_type,
                                    policy_type=PolicyType.UCB,
                                    policy_args=policy_args,
                                    assign_priority=assign_priority)})

                            new_row = {
                                'batch_size': batch_size,
                                'algorithm_type': algorithm_type,
                                'optimizer': optimizer_type.name.lower(),
                                'learning_rate': learning_rate,
                                'is_double': 'Yes' if is_double else 'No',
                                'assign_priority': 'Yes' if assign_priority else 'No',
                            }

                            result = run_pettingzoo_env(env, agents, n_games_train=500, n_games_test=50)

                            for key in result:
                                new_row.update({key: result[key]})

                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_all_td_methods(env, env_name):
    print('Running', env_name)
    run_td_epsilon_greedy(env, env_name)
    run_dueling_td_epsilon_greedy(env, env_name)
    run_td_softmax(env, env_name)
    run_dueling_td_softmax(env, env_name)
    run_td_ucb(env, env_name)
    run_dueling_td_ucb(env, env_name)


def run_ddpg(env, env_name):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_ddpg.csv'.format(env_name))

    result_cols = ['batch_size', 'actor_learning_rate',
                   'critic_learning_rate', 'tau', 'assign_priority',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)
    network_args = {
        'fc_dim': (1024, 512),
        'cnn_dims': [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    }

    for batch_size in [64, 128]:
        for actor_learning_rate in [0.001, 0.0001]:
            for critic_learning_rate in [0.001, 0.0001]:
                for tau in [1e-2, 1e-3]:
                    for assign_priority in [False, True]:
                        actor_optimizer_args = {
                            'learning_rate': actor_learning_rate
                        }
                        critic_optimizer_args = {
                            'learning_rate': critic_learning_rate
                        }
                        agents = {}

                        for agent_id in env.possible_agents:
                            agents.update({agent_id: DDPGAgent(
                                input_dims=env.observation_spaces[agent_id].shape,
                                action_space=env.action_spaces[agent_id],
                                tau=tau, network_args=network_args,
                                batch_size=batch_size,
                                actor_optimizer_type=NetworkOptimizer.ADAM,
                                critic_optimizer_type=NetworkOptimizer.ADAM,
                                actor_optimizer_args=actor_optimizer_args,
                                critic_optimizer_args=critic_optimizer_args,
                                assign_priority=assign_priority
                            )})

                        result = run_pettingzoo_env(env, agents, n_games_train=500, n_games_test=50)

                        new_row = {
                            'batch_size': batch_size,
                            'actor_learning_rate': actor_learning_rate,
                            'critic_learning_rate': critic_learning_rate,
                            'tau': tau,
                            'assign_priority': 'Yes' if assign_priority else 'No'
                        }

                        for key in result:
                            new_row.update({key: result[key]})

                        results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td3(env, env_name):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_td3.csv'.format(env_name))

    result_cols = ['batch_size', 'actor_learning_rate',
                   'critic_learning_rate', 'tau', 'assign_priority',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    actor_optimizer_args = {
        'learning_rate': 1e-3
    }
    critic_optimizer_args = {
        'learning_rate': 1e-3
    }
    network_args = {
        'fc_dim': (1024, 512),
        'cnn_dims': [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    }

    for batch_size in [64, 100, 128]:
        for tau in [0.005, 0.01]:
            for assign_priority in [False, True]:
                agents = {}

                for agent_id in env.possible_agents:
                    agents.update({agent_id: TD3Agent(
                        input_dims=env.observation_spaces[agent_id].shape,
                        action_space=env.action_spaces[agent_id],
                        tau=tau, network_args=network_args,
                        batch_size=batch_size,
                        actor_optimizer_type=NetworkOptimizer.ADAM,
                        critic_optimizer_type=NetworkOptimizer.ADAM,
                        actor_optimizer_args=actor_optimizer_args,
                        critic_optimizer_args=critic_optimizer_args,
                        assign_priority=assign_priority
                    )})

                result = run_pettingzoo_env(env, agents, n_games_train=500, n_games_test=50)

                new_row = {
                    'batch_size': batch_size,
                    'tau': tau,
                    'assign_priority': 'Yes' if assign_priority else 'No'
                }

                for key in result:
                    new_row.update({key: result[key]})

                results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_actor_critic_continuous_methods(env, env_name):
    print('Running', env_name)
    run_ddpg(env, env_name)
    run_td3(env, env_name)


def run_decision_tree_heuristics(env, env_name, heuristic_func, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_dt.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for learning_type in [LearningType.OFFLINE]:
        for use_model_only in [False, True]:

            agents = {}

            for agent_id in env.possible_agents:
                agents.update({agent_id: HeuristicWithDT(heuristic_func, use_model_only, env.action_spaces[agent_id],
                                                         False, 0, False, None, None, None, **args)})

            result = run_pettingzoo_env(env, agents, learning_type=learning_type,
                                        n_games_train=500, n_games_test=50)

            new_row = {
                'learning_type': learning_type.name,
                'use_model_only': 'Yes' if use_model_only else 'No'
            }

            for key in result:
                new_row.update({key: result[key]})

            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, float_format='%.3f', index=False)


def run_td_epsilon_greedy_heuristics(env, env_name, heuristic_func, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_td_epsilon_greedy.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size',
                   'optimizer', 'learning_rate', 'is_double', 'algorithm_type', 'enable_decay',
                   'epsilon_start', 'add_conservative_loss', 'alpha', 'num_time_steps_test',
                   'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}
    network_args = {
        'fc_dim': 512,
        'cnn_dims': [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    }

    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for is_double in [False, True]:
                for algorithm_type in TDAlgorithmType.all():
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                for add_conservative_loss in [False, True]:
                                    for alpha in [0.001]:
                                        for enable_decay in [False, True]:
                                            epsilons = [1.0] if enable_decay \
                                                else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                                            for epsilon in epsilons:
                                                policy_args.update(
                                                    {'eps_start': epsilon,
                                                     'enable_decay': enable_decay})
                                                network_optimizer_args = {
                                                    'learning_rate': learning_rate
                                                }
                                                agents = {}

                                                for agent_id in env.possible_agents:
                                                    agents.update({agent_id: HeuristicWithTD(
                                                        input_dims=env.observation_spaces[agent_id].shape,
                                                        action_space=env.action_spaces[agent_id],
                                                        gamma=0.99,
                                                        mem_size=1000000,
                                                        batch_size=batch_size,
                                                        network_args=network_args,
                                                        optimizer_type=optimizer_type,
                                                        replace=1000,
                                                        optimizer_args=network_optimizer_args,
                                                        is_double=is_double,
                                                        algorithm_type=algorithm_type,
                                                        policy_type=PolicyType.EPSILON_GREEDY,
                                                        policy_args=policy_args,
                                                        use_model_only=use_model_only,
                                                        learning_type=learning_type,
                                                        heuristic_func=heuristic_func,
                                                        add_conservative_loss=add_conservative_loss,
                                                        alpha=alpha,
                                                        **args)})

                                                result = run_pettingzoo_env(env, agents,
                                                                            learning_type=learning_type,
                                                                            n_games_train=500, n_games_test=50)

                                                new_row = {
                                                    'batch_size': batch_size,
                                                    'algorithm_type': algorithm_type,
                                                    'optimizer': optimizer_type.name.lower(),
                                                    'learning_rate': learning_rate,
                                                    'is_double': 'Yes' if is_double else 'No',
                                                    'enable_decay': 'Yes' if enable_decay else 'No',
                                                    'epsilon': epsilon,
                                                    'learning_type': learning_type.name,
                                                    'use_model_only': 'Yes' if use_model_only else 'No',
                                                    'add_conservative_loss': 'Yes' if add_conservative_loss else 'No',
                                                    'alpha': alpha if add_conservative_loss else 0,
                                                }
                                                for key in result:
                                                    new_row.update({key: result[key]})

                                                results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td_softmax_heuristics(env, env_name, heuristic_func, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_td_softmax.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size',
                   'optimizer', 'learning_rate', 'is_double', 'algorithm_type', 'tau',
                   'add_conservative_loss', 'alpha',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}
    network_args = {
        'fc_dim': 512,
        'cnn_dims': [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    }
    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for is_double in [False, True]:
                for algorithm_type in TDAlgorithmType.all():
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                for add_conservative_loss in [False, True]:
                                    for alpha in [0.001]:
                                        for tau in [0.0001, 0.001, 0.1, 1.0, 10.0]:
                                            policy_args.update({'tau': tau})
                                            network_optimizer_args = {
                                                'learning_rate': learning_rate
                                            }
                                            agents = {}

                                            for agent_id in env.possible_agents:
                                                agents.update({agent_id: HeuristicWithTD(
                                                    input_dims=env.observation_spaces[agent_id].shape,
                                                    action_space=env.action_spaces[agent_id],
                                                    gamma=0.99,
                                                    mem_size=1000000,
                                                    batch_size=batch_size,
                                                    network_args=network_args,
                                                    optimizer_type=optimizer_type,
                                                    replace=1000,
                                                    optimizer_args=network_optimizer_args,
                                                    is_double=is_double,
                                                    algorithm_type=algorithm_type,
                                                    policy_type=PolicyType.SOFTMAX,
                                                    policy_args=policy_args,
                                                    heuristic_func=heuristic_func,
                                                    learning_type=learning_type,
                                                    add_conservative_loss=add_conservative_loss,
                                                    alpha=alpha,
                                                    use_model_only=use_model_only,
                                                    **args)})

                                            result = run_pettingzoo_env(env, agents,
                                                                        learning_type=learning_type,
                                                                        n_games_train=500, n_games_test=50)

                                            new_row = {
                                                'batch_size': batch_size,
                                                'algorithm_type': algorithm_type,
                                                'optimizer': optimizer_type.name.lower(),
                                                'learning_rate': learning_rate,
                                                'is_double': 'Yes' if is_double else 'No',
                                                'tau': tau,
                                                'use_model_only': 'Yes' if use_model_only else 'No',
                                                'learning_type': learning_type.name,
                                                'add_conservative_loss': 'Yes' if add_conservative_loss else 'No',
                                                'alpha': alpha if add_conservative_loss else 0
                                            }
                                            for key in result:
                                                new_row.update({key: result[key]})

                                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td_ucb_heuristics(env, env_name, heuristic_func, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_td_ucb.csv'.format(env_name))

    policy_args = {'confidence_factor': 2}
    network_args = {
        'fc_dim': 512,
        'cnn_dims': [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    }

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'optimizer', 'learning_rate',
                   'is_double', 'algorithm_type', 'add_conservative_loss', 'alpha',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for is_double in [False, True]:
                for algorithm_type in TDAlgorithmType.all():
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                for add_conservative_loss in [False, True]:
                                    for alpha in [0.001]:
                                        network_optimizer_args = {
                                            'learning_rate': learning_rate
                                        }
                                        agents = {}

                                        for agent_id in env.possible_agents:
                                            agents.update({agent_id: HeuristicWithTD(
                                                input_dims=env.observation_spaces[agent_id].shape,
                                                action_space=env.action_spaces[agent_id],
                                                gamma=0.99,
                                                mem_size=1000000,
                                                batch_size=batch_size,
                                                network_args=network_args,
                                                optimizer_type=optimizer_type,
                                                replace=1000,
                                                optimizer_args=network_optimizer_args,
                                                is_double=is_double,
                                                algorithm_type=algorithm_type,
                                                policy_type=PolicyType.UCB,
                                                policy_args=policy_args,
                                                learning_type=learning_type,
                                                use_model_only=use_model_only,
                                                heuristic_func=heuristic_func,
                                                add_conservative_loss=add_conservative_loss,
                                                alpha=alpha, **args)})

                                        result = run_pettingzoo_env(env, agents, learning_type=learning_type,
                                                                    n_games_train=500, n_games_test=50)

                                        new_row = {
                                            'batch_size': batch_size,
                                            'algorithm_type': algorithm_type,
                                            'optimizer': optimizer_type.name.lower(),
                                            'learning_rate': learning_rate,
                                            'is_double': 'Yes' if is_double else 'No',
                                            'learning_type': learning_type.name,
                                            'use_model_only': 'Yes' if use_model_only else 'No',
                                            'add_conservative_loss': 'Yes' if add_conservative_loss else 'No',
                                            'alpha': alpha if add_conservative_loss else 0,
                                        }
                                        for key in result:
                                            new_row.update({key: result[key]})

                                        results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_epsilon_greedy_heuristics(env, env_name, heuristic_func, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_dueling_td_epsilon_greedy.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'optimizer', 'learning_rate',
                   'is_double', 'algorithm_type', 'enable_decay', 'epsilon_start',
                   'add_conservative_loss', 'alpha', 'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}
    network_args = {
        'fc_dim': (1024, 512),
        'cnn_dims': [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    }

    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for is_double in [False, True]:
                for algorithm_type in TDAlgorithmType.all():
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                for add_conservative_loss in [False, True]:
                                    for alpha in [0.001]:
                                        for enable_decay in [False, True]:
                                            epsilons = [1.0] if enable_decay \
                                                else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                                            for epsilon in epsilons:
                                                policy_args.update(
                                                    {'eps_start': epsilon,
                                                     'enable_decay': enable_decay})
                                                network_optimizer_args = {
                                                    'learning_rate': learning_rate
                                                }
                                                agents = {}

                                                for agent_id in env.possible_agents:
                                                    agents.update({agent_id: HeuristicWithDuelingTD(
                                                        input_dims=env.observation_spaces[agent_id].shape,
                                                        action_space=env.action_spaces[agent_id],
                                                        gamma=0.99,
                                                        mem_size=1000000,
                                                        batch_size=batch_size,
                                                        network_args=network_args,
                                                        optimizer_type=optimizer_type,
                                                        replace=1000,
                                                        optimizer_args=network_optimizer_args,
                                                        is_double=is_double,
                                                        algorithm_type=algorithm_type,
                                                        policy_type=PolicyType.EPSILON_GREEDY,
                                                        policy_args=policy_args,
                                                        use_model_only=use_model_only,
                                                        learning_type=learning_type,
                                                        heuristic_func=heuristic_func,
                                                        add_conservative_loss=add_conservative_loss,
                                                        alpha=alpha, **args)})

                                                result = run_pettingzoo_env(env, agents,
                                                                            learning_type=learning_type,
                                                                            n_games_train=500, n_games_test=50)

                                                new_row = {
                                                    'batch_size': batch_size,
                                                    'algorithm_type': algorithm_type,
                                                    'optimizer': optimizer_type.name.lower(),
                                                    'learning_rate': learning_rate,
                                                    'is_double': 'Yes' if is_double else 'No',
                                                    'enable_decay': 'Yes' if enable_decay else 'No',
                                                    'epsilon': epsilon,
                                                    'learning_type': learning_type.name,
                                                    'use_model_only': 'Yes' if use_model_only else 'No',
                                                    'add_conservative_loss': 'Yes' if add_conservative_loss else 'No',
                                                    'alpha': alpha if add_conservative_loss else 0,
                                                }
                                                for key in result:
                                                    new_row.update({key: result[key]})

                                                results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_softmax_heuristics(env, env_name, heuristic_func, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_dueling_td_softmax.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size',
                   'optimizer', 'learning_rate', 'is_double', 'algorithm_type', 'tau',
                   'add_conservative_loss', 'alpha',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}
    network_args = {
        'fc_dim': (1024, 512),
        'cnn_dims': [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    }

    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for is_double in [False, True]:
                for algorithm_type in TDAlgorithmType.all():
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                for add_conservative_loss in [False, True]:
                                    for alpha in [0.001]:
                                        for tau in [0.0001, 0.001, 0.1, 1.0, 10.0]:
                                            policy_args.update({'tau': tau})
                                            network_optimizer_args = {
                                                'learning_rate': learning_rate
                                            }
                                            agents = {}

                                            for agent_id in env.possible_agents:
                                                agents.update({agent_id: HeuristicWithDuelingTD(
                                                    input_dims=env.observation_spaces[agent_id].shape,
                                                    action_space=env.action_spaces[agent_id],
                                                    gamma=0.99,
                                                    mem_size=1000000,
                                                    batch_size=batch_size,
                                                    network_args=network_args,
                                                    optimizer_type=optimizer_type,
                                                    replace=1000,
                                                    optimizer_args=network_optimizer_args,
                                                    is_double=is_double,
                                                    algorithm_type=algorithm_type,
                                                    policy_type=PolicyType.SOFTMAX,
                                                    policy_args=policy_args,
                                                    heuristic_func=heuristic_func,
                                                    learning_type=learning_type,
                                                    add_conservative_loss=add_conservative_loss,
                                                    alpha=alpha,
                                                    use_model_only=use_model_only, **args)})

                                            result = run_pettingzoo_env(env, agents,
                                                                        learning_type=learning_type,
                                                                        n_games_train=500, n_games_test=50)

                                            new_row = {
                                                'batch_size': batch_size,
                                                'algorithm_type': algorithm_type,
                                                'optimizer': optimizer_type.name.lower(),
                                                'learning_rate': learning_rate,
                                                'is_double': 'Yes' if is_double else 'No',
                                                'tau': tau,
                                                'use_model_only': 'Yes' if use_model_only else 'No',
                                                'learning_type': learning_type.name,
                                                'add_conservative_loss': 'Yes' if add_conservative_loss else 'No',
                                                'alpha': alpha if add_conservative_loss else 0,
                                            }
                                            for key in result:
                                                new_row.update({key: result[key]})

                                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_ucb_heuristics(env, env_name, heuristic_func, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_dueling_td_ucb.csv'.format(env_name))

    policy_args = {'confidence_factor': 2}

    network_args = {
        'fc_dim': (1024, 512),
        'cnn_dims': [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    }

    result_cols = ['learning_type', 'use_model_only', 'batch_size',
                   'optimizer', 'learning_rate',
                   'is_double', 'algorithm_type', 'add_conservative_loss', 'alpha',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for is_double in [False, True]:
                for algorithm_type in TDAlgorithmType.all():
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                for add_conservative_loss in [False, True]:
                                    for alpha in [0.001]:
                                        network_optimizer_args = {
                                            'learning_rate': learning_rate
                                        }
                                        agents = {}

                                        for agent_id in env.possible_agents:
                                            agents.update({agent_id: HeuristicWithDuelingTD(
                                                input_dims=env.observation_spaces[agent_id].shape,
                                                action_space=env.action_spaces[agent_id],
                                                gamma=0.99,
                                                mem_size=1000000,
                                                batch_size=batch_size,
                                                network_args=network_args,
                                                optimizer_type=optimizer_type,
                                                replace=1000,
                                                optimizer_args=network_optimizer_args,
                                                is_double=is_double,
                                                algorithm_type=algorithm_type,
                                                policy_type=PolicyType.UCB,
                                                policy_args=policy_args,
                                                learning_type=learning_type,
                                                use_model_only=use_model_only,
                                                heuristic_func=heuristic_func,
                                                add_conservative_loss=add_conservative_loss,
                                                alpha=alpha, **args)})

                                        result = run_pettingzoo_env(env, agents, learning_type=learning_type,
                                                                    n_games_train=500, n_games_test=50)

                                        new_row = {
                                            'batch_size': batch_size,
                                            'algorithm_type': algorithm_type,
                                            'optimizer': optimizer_type.name.lower(),
                                            'learning_rate': learning_rate,
                                            'is_double': 'Yes' if is_double else 'No',
                                            'learning_type': learning_type.name,
                                            'use_model_only': 'Yes' if use_model_only else 'No',
                                            'add_conservative_loss': 'Yes' if add_conservative_loss else 'No',
                                            'alpha': alpha if add_conservative_loss else 0
                                        }
                                        for key in result:
                                            new_row.update({key: result[key]})

                                        results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_ddpg_heuristics(env, env_name, heuristic_func, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristics_ddpg.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'actor_learning_rate',
                   'critic_learning_rate', 'tau',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)
    network_args = {
        'fc_dim': (1024, 512),
        'cnn_dims': [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    }

    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for batch_size in [64, 128]:
                for actor_learning_rate in [0.001, 0.0001]:
                    for critic_learning_rate in [0.001, 0.0001]:
                        for tau in [1e-2, 1e-3]:
                            actor_optimizer_args = {
                                'learning_rate': actor_learning_rate
                            }
                            critic_optimizer_args = {
                                'learning_rate': critic_learning_rate
                            }
                            agents = {}

                            for agent_id in env.possible_agents:
                                agents.update({agent_id: HeuristicWithDDPG(
                                    input_dims=env.observation_spaces[agent_id].shape,
                                    action_space=env.action_spaces[agent_id],
                                    tau=tau, network_args=network_args,
                                    batch_size=batch_size,
                                    actor_optimizer_type=NetworkOptimizer.ADAM,
                                    critic_optimizer_type=NetworkOptimizer.ADAM,
                                    actor_optimizer_args=actor_optimizer_args,
                                    critic_optimizer_args=critic_optimizer_args,
                                    learning_type=learning_type,
                                    use_model_only=use_model_only,
                                    heuristic_func=heuristic_func, **args
                                )})

                            result = run_pettingzoo_env(env, agents, learning_type=learning_type,
                                                        n_games_train=500, n_games_test=50)

                            new_row = {
                                'learning_type': learning_type.name,
                                'use_model_only': 'Yes' if use_model_only else 'No',
                                'batch_size': batch_size,
                                'actor_learning_rate': actor_learning_rate,
                                'critic_learning_rate': critic_learning_rate,
                                'tau': tau
                            }

                            for key in result:
                                new_row.update({key: result[key]})

                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td3_heuristics(env, env_name, heuristic_func, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristics_td3.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'actor_learning_rate',
                   'critic_learning_rate', 'tau',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    network_args = {
        'fc_dim': (1024, 512),
        'cnn_dims': [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    }

    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for batch_size in [64, 128]:
                for actor_learning_rate in [0.001, 0.0001]:
                    for critic_learning_rate in [0.001, 0.0001]:
                        for tau in [1e-2, 1e-3]:
                            actor_optimizer_args = {
                                'learning_rate': actor_learning_rate
                            }
                            critic_optimizer_args = {
                                'learning_rate': critic_learning_rate
                            }
                            agents = {}

                            for agent_id in env.possible_agents:
                                agents.update({agent_id: HeuristicWithTD3(
                                    input_dims=env.observation_spaces[agent_id].shape,
                                    action_space=env.action_spaces[agent_id],
                                    tau=tau, network_args=network_args,
                                    batch_size=batch_size,
                                    actor_optimizer_type=NetworkOptimizer.ADAM,
                                    critic_optimizer_type=NetworkOptimizer.ADAM,
                                    actor_optimizer_args=actor_optimizer_args,
                                    critic_optimizer_args=critic_optimizer_args,
                                    learning_type=learning_type,
                                    use_model_only=use_model_only,
                                    heuristic_func=heuristic_func, **args
                                )})

                            result = run_pettingzoo_env(env, agents, learning_type=learning_type,
                                                        n_games_train=500, n_games_test=50)

                            new_row = {
                                'learning_type': learning_type.name,
                                'use_model_only': 'Yes' if use_model_only else 'No',
                                'batch_size': batch_size,
                                'actor_learning_rate': actor_learning_rate,
                                'critic_learning_rate': critic_learning_rate,
                                'tau': tau
                            }

                            for key in result:
                                new_row.update({key: result[key]})

                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_cem_heuristics(env, env_name, heuristic_func, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_cem.csv'.format(env_name))

    result_cols = ['use_model_only', 'learning_type',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    network_args = {
        'fc_dim': (1024, 512),
        'cnn_dims': [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    }

    for use_model_only in [False, True]:
        for learning_type in [LearningType.OFFLINE, LearningType.OFFLINE, LearningType.BOTH]:
            agent = HeuristicWithCEM(input_dims=env.observation_space.shape, action_space=env.action_shape,
                                     network_args=network_args, use_model_only=use_model_only,
                                     learning_type=learning_type, heuristic_func=heuristic_func, **args)

            new_row = {'use_model_only': 'Yes' if use_model_only else 'No',
                       'learning_type': learning_type.name}

            result = run_pettingzoo_env(env, agent, n_games_train=500, n_games_test=50, learning_type=learning_type)

            for key in result:
                new_row.update({key: result[key]})

            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, float_format='%.3f', index=False)


def run_discrete_heuristics(env, env_name, heuristic_func, **args):
    run_decision_tree_heuristics(env, env_name, heuristic_func, **args)
    run_td_epsilon_greedy_heuristics(env, env_name, heuristic_func, **args)
    run_dueling_td_epsilon_greedy_heuristics(env, env_name, heuristic_func, **args)
    run_td_softmax_heuristics(env, env_name, heuristic_func, **args)
    run_dueling_td_softmax_heuristics(env, env_name, heuristic_func, **args)
    run_td_ucb_heuristics(env, env_name, heuristic_func, **args)
    run_dueling_td_ucb_heuristics(env, env_name, heuristic_func, **args)


def run_continuous_heuristics(env, env_name, heuristic_func, **args):
    run_decision_tree_heuristics(env, env_name, heuristic_func, **args)
    run_ddpg_heuristics(env, env_name, heuristic_func, **args)
    run_td3_heuristics(env, env_name, heuristic_func, **args)


continuous_env = pistonball_v4.env(continuous=True)
discrete_env = pistonball_v4.env(continuous=False)


def continuous_policy(self, obs):
    DOWN = np.array([-1.0])
    UP = np.array([1.0])
    GRAYSCALE_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    obs = (obs.astype(np.float32) @ GRAYSCALE_WEIGHTS).astype(np.uint8)
    ball_vals = np.equal(obs, 137)
    first_loc = np.argmax(ball_vals, axis=1)
    if first_loc.any():
        first_loc_nonzero = np.where(first_loc == 0, 1000, first_loc)
        min_loc = np.min(first_loc_nonzero)
        max_loc = np.max(first_loc)
        if min_loc < 5:
            return UP
        elif max_loc > 80:
            return DOWN

    first_piston_vals = np.equal(obs, 73)
    pi1 = 200 - np.argmax(first_piston_vals[:, 11])
    pi2 = 200 - np.argmax(first_piston_vals[:, 51])
    pi3 = 200 - np.argmax(first_piston_vals[:, 91])
    if pi1 == 200:
        action = DOWN
    elif pi3 == 200:
        action = UP
    else:
        if pi2 > pi3:
            action = DOWN
        elif pi1 > pi2:
            action = UP
        elif pi1 + 1 < pi2:
            action = DOWN
        elif pi2 + 16 < pi3:
            action = UP
        else:
            action = UP

    return action


def discrete_policy(self, obs):
    DOWN = 0
    UP = 2
    GRAYSCALE_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    obs = (obs.astype(np.float32) @ GRAYSCALE_WEIGHTS).astype(np.uint8)
    ball_vals = np.equal(obs, 137)
    first_loc = np.argmax(ball_vals, axis=1)
    if first_loc.any():
        first_loc_nonzero = np.where(first_loc == 0, 1000, first_loc)
        min_loc = np.min(first_loc_nonzero)
        max_loc = np.max(first_loc)
        if min_loc < 5:
            return UP
        elif max_loc > 80:
            return DOWN

    first_piston_vals = np.equal(obs, 73)
    pi1 = 200 - np.argmax(first_piston_vals[:, 11])
    pi2 = 200 - np.argmax(first_piston_vals[:, 51])
    pi3 = 200 - np.argmax(first_piston_vals[:, 91])
    if pi1 == 200:
        action = DOWN
    elif pi3 == 200:
        action = UP
    else:
        if pi2 > pi3:
            action = DOWN
        elif pi1 > pi2:
            action = UP
        elif pi1 + 1 < pi2:
            action = DOWN
        elif pi2 + 16 < pi3:
            action = UP
        else:
            action = UP

    return action


run_all_td_methods(discrete_env, 'pistonball')
run_actor_critic_continuous_methods(continuous_env, 'pistonball')
run_discrete_heuristics(discrete_env, 'pistonball', discrete_policy)
run_continuous_heuristics(continuous_env, 'pistonball', continuous_policy)
