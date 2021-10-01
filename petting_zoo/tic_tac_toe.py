import os

import pandas as pd
from gym.spaces import Discrete
from torch_rl.heuristic.heuristic_with_dt import HeuristicWithDT
from torch_rl.heuristic.heuristic_with_rf import HeuristicWithRF
from torch_rl.heuristic.heuristic_with_td import HeuristicWithTD
from torch_rl.heuristic.heuristic_with_td3 import HeuristicWithTD3
from torch_rl.heuristic.heuristic_with_dueling_td import HeuristicWithDuelingTD
from torch_rl.heuristic.heuristic_with_ddpg import HeuristicWithDDPG
from torch_rl.heuristic.heuristic_with_hill_climbing import HeuristicWithHillClimbing
from torch_rl.heuristic.heuristic_with_cem import HeuristicWithCEM
from torch_rl.td.agent import TDAgent
from torch_rl.dueling_td.agent import DuelingTDAgent
from torch_rl.ddpg.agent import DDPGAgent
from torch_rl.td3.agent import TD3Agent
from torch_rl.hill_climbing.agent import HillClimbingAgent
from torch_rl.cem.agent import CEMAgent
from torch_rl.utils.types import NetworkOptimizer, TDAlgorithmType, PolicyType, LearningType

from run import run_gym_env
from utils import develop_memory_from_gym_env, derive_hidden_layer_size

max_time_steps = 500 * 1000


def run_td_epsilon_greedy(env, env_name, penalty, env_goal=None):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_td_epsilon_greedy.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate', 'goal_focused',
                   'assign_priority', 'enable_action_blocker', 'action_blocker_model_type', 'use_preloaded_memory',
                   'is_double', 'algorithm_type', 'enable_decay', 'epsilon_start', 'use_mse',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    pre_loaded_memory = develop_memory_from_gym_env(env, max_time_steps)

    policy_args = {}

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in list({False, penalty > 0}):
                action_blocker_model_types = ['decision_tree', 'random_forest'] if enable_action_blocker else [None]
                for action_blocker_model_type in action_blocker_model_types:
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                               64, 128, 256, 512}):
                                    for goal in list({None, env_goal}):
                                        for enable_decay in [False, True]:
                                            epsilons = [1.0] if enable_decay \
                                                else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                                            for epsilon in epsilons:
                                                policy_args.update({'eps_start': epsilon, 'enable_decay': enable_decay})
                                                for assign_priority in [False, True]:
                                                    for use_preloaded_memory in [False, True]:
                                                        for use_mse in [False, True]:
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
                                                                goal=goal,
                                                                is_double=is_double,
                                                                algorithm_type=algorithm_type,
                                                                policy_type=PolicyType.EPSILON_GREEDY,
                                                                policy_args=policy_args,
                                                                assign_priority=assign_priority,
                                                                pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                                action_blocker_timesteps=max_time_steps,
                                                                action_blocker_model_type=action_blocker_model_type,
                                                                use_mse=use_mse)

                                                            result = run_gym_env(env, agent, n_games_train=500,
                                                                                 n_games_test=50)

                                                            new_row = {
                                                                'batch_size': batch_size,
                                                                'hidden_layer_size': hidden_layer_size,
                                                                'algorithm_type': algorithm_type,
                                                                'optimizer': optimizer_type.name.lower(),
                                                                'learning_rate': learning_rate,
                                                                'goal_focused': 'Yes' if goal else 'No',
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


def run_td_softmax(env, env_name, penalty, env_goal=None):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_td_softmax.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate', 'goal_focused',
                   'assign_priority', 'enable_action_blocker', 'action_blocker_model_type', 'use_preloaded_memory',
                   'is_double', 'algorithm_type', 'tau', 'use_mse',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    pre_loaded_memory = develop_memory_from_gym_env(env, max_time_steps)
    policy_args = {}

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in list({False, penalty > 0}):
                action_blocker_model_types = ['decision_tree', 'random_forest'] if enable_action_blocker else [None]
                for action_blocker_model_type in action_blocker_model_types:
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                               64, 128, 256, 512}):
                                    for goal in list({None, env_goal}):
                                        for tau in [0.0001, 0.001, 0.1, 1.0, 10.0]:
                                            policy_args.update({'tau': tau})
                                            for assign_priority in [False, True]:
                                                for use_preloaded_memory in [False, True]:
                                                    for use_mse in [False, True]:
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
                                                            goal=goal,
                                                            is_double=is_double,
                                                            algorithm_type=algorithm_type,
                                                            policy_type=PolicyType.SOFTMAX,
                                                            policy_args=policy_args,
                                                            assign_priority=assign_priority,
                                                            pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                            action_blocker_model_type=action_blocker_model_type,
                                                            action_blocker_timesteps=max_time_steps,
                                                            use_mse=use_mse)

                                                        result = run_gym_env(env, agent, n_games_train=500, n_games_test=50)

                                                        new_row = {
                                                            'batch_size': batch_size,
                                                            'hidden_layer_size': hidden_layer_size,
                                                            'algorithm_type': algorithm_type,
                                                            'optimizer': optimizer_type.name.lower(),
                                                            'learning_rate': learning_rate,
                                                            'goal_focused': 'Yes' if goal else 'No',
                                                            'is_double': 'Yes' if is_double else 'No',
                                                            'tau': tau,
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


def run_td_ucb(env, env_name, penalty, env_goal=None):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results', '{0}_td_ucb.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate', 'goal_focused',
                   'assign_priority', 'is_double', 'algorithm_type',
                   'enable_action_blocker', 'action_blocker_model_type', 'use_preloaded_memory', 'use_mse',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    pre_loaded_memory = develop_memory_from_gym_env(env, max_time_steps)

    policy_args = {'confidence_factor': 2}

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in list({False, penalty > 0}):
                action_blocker_model_types = ['decision_tree', 'random_forest'] if enable_action_blocker else [None]
                for action_blocker_model_type in action_blocker_model_types:
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                               64, 128, 256, 512}):
                                    for goal in list({None, env_goal}):
                                        for assign_priority in [False, True]:
                                            for use_preloaded_memory in [False, True]:
                                                for use_mse in [False, True]:
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
                                                        goal=goal,
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

                                                    result = run_gym_env(env, agent, n_games_train=500, n_games_test=50)

                                                    new_row = {
                                                        'batch_size': batch_size,
                                                        'hidden_layer_size': hidden_layer_size,
                                                        'algorithm_type': algorithm_type,
                                                        'optimizer': optimizer_type.name.lower(),
                                                        'learning_rate': learning_rate,
                                                        'goal_focused': 'Yes' if goal else 'No',
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


def run_td_thompson_sampling(env, env_name, penalty, env_goal=None):
    assert penalty > 0
    pre_loaded_memory = develop_memory_from_gym_env(env, max_time_steps)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_td_thompson_sampling.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate', 'goal_focused',
                   'assign_priority', 'is_double', 'algorithm_type',
                   'enable_action_blocker', 'action_blocker_model_type', 'use_preloaded_memory', 'use_mse',
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

                                for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                               64, 128, 256, 512}):
                                    for goal in list({None, env_goal}):
                                        for assign_priority in [False, True]:
                                            for use_preloaded_memory in [False, True]:
                                                for use_mse in [False, True]:
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
                                                        goal=goal,
                                                        is_double=is_double,
                                                        algorithm_type=algorithm_type,
                                                        policy_type=PolicyType.THOMPSON_SAMPLING,
                                                        policy_args=policy_args,
                                                        assign_priority=assign_priority,
                                                        action_blocker_model_type=action_blocker_model_type,
                                                        action_blocker_timesteps=max_time_steps,
                                                        pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                        use_mse=use_mse)

                                                    result = run_gym_env(env, agent, n_games_train=500, n_games_test=50)

                                                    new_row = {
                                                        'batch_size': batch_size,
                                                        'hidden_layer_size': hidden_layer_size,
                                                        'algorithm_type': algorithm_type,
                                                        'optimizer': optimizer_type.name.lower(),
                                                        'learning_rate': learning_rate,
                                                        'goal_focused': 'Yes' if goal else 'No',
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


def run_dueling_td_epsilon_greedy(env, env_name, penalty, env_goal=None):
    pre_loaded_memory = develop_memory_from_gym_env(env, max_time_steps)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_td_epsilon_greedy.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate', 'goal_focused',
                   'is_double', 'algorithm_type', 'enable_decay',
                   'epsilon_start', 'enable_action_blocker', 'action_blocker_model_type', 'use_preloaded_memory',
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
                                for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                               64, 128, 256, 512}):
                                    for goal in list({None, env_goal}):
                                        for enable_decay in [False, True]:
                                            epsilons = [1.0] if enable_decay \
                                                else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                                            for epsilon in epsilons:
                                                policy_args.update({'eps_start': epsilon, 'enable_decay': enable_decay})
                                                for assign_priority in [False, True]:
                                                    for use_preloaded_memory in [False, True]:
                                                        for use_mse in [False, True]:
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
                                                                goal=goal,
                                                                is_double=is_double,
                                                                algorithm_type=algorithm_type,
                                                                policy_type=PolicyType.EPSILON_GREEDY,
                                                                policy_args=policy_args,
                                                                assign_priority=assign_priority,
                                                                action_blocker_model_type=action_blocker_model_type,
                                                                action_blocker_timesteps=max_time_steps,
                                                                pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                                use_mse=use_mse)

                                                            result = run_gym_env(env, agent, n_games_train=500,
                                                                                 n_games_test=50)

                                                            new_row = {
                                                                'batch_size': batch_size,
                                                                'hidden_layer_size': hidden_layer_size,
                                                                'algorithm_type': algorithm_type,
                                                                'optimizer': optimizer_type.name.lower(),
                                                                'learning_rate': learning_rate,
                                                                'goal_focused': 'Yes' if goal else 'No',
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


def run_dueling_td_softmax(env, env_name, penalty, env_goal=None):
    pre_loaded_memory = develop_memory_from_gym_env(env, max_time_steps)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_td_softmax.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate', 'goal_focused',
                   'assign_priority', 'is_double', 'algorithm_type', 'tau',
                   'enable_action_blocker', 'action_blocker_model_type', 'use_preloaded_memory', 'use_mse',
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
                                for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                               64, 128, 256, 512}):
                                    for goal in list({None, env_goal}):
                                        for tau in [0.0001, 0.001, 0.1, 1.0, 10.0]:
                                            policy_args.update({'tau': tau})
                                            for assign_priority in [False, True]:
                                                for use_preloaded_memory in [False, True]:
                                                    for use_mse in [False, True]:
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
                                                            goal=goal,
                                                            is_double=is_double,
                                                            algorithm_type=algorithm_type,
                                                            policy_type=PolicyType.SOFTMAX,
                                                            policy_args=policy_args,
                                                            assign_priority=assign_priority,
                                                            action_blocker_model_type=action_blocker_model_type,
                                                            action_blocker_timesteps=max_time_steps,
                                                            pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                            use_mse=use_mse)

                                                        result = run_gym_env(env, agent, n_games_train=500, n_games_test=50)

                                                        new_row = {
                                                            'batch_size': batch_size,
                                                            'hidden_layer_size': hidden_layer_size,
                                                            'algorithm_type': algorithm_type,
                                                            'optimizer': optimizer_type.name.lower(),
                                                            'learning_rate': learning_rate,
                                                            'goal_focused': 'Yes' if goal else 'No',
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


def run_dueling_td_ucb(env, env_name, penalty, env_goal=None):
    pre_loaded_memory = develop_memory_from_gym_env(env, max_time_steps)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_td_ucb.csv'.format(env_name))
    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate', 'goal_focused',
                   'assign_priority', 'is_double', 'algorithm_type',
                   'enable_action_blocker', 'action_blocker_model_type', 'use_preloaded_memory', 'use_mse',
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
                                for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                               64, 128, 256, 512}):
                                    for goal in list({None, env_goal}):
                                        for assign_priority in [False, True]:
                                            for use_preloaded_memory in [False, True]:
                                                for use_mse in [False, True]:
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
                                                        goal=goal,
                                                        is_double=is_double,
                                                        algorithm_type=algorithm_type,
                                                        policy_type=PolicyType.UCB,
                                                        policy_args=policy_args,
                                                        assign_priority=assign_priority,
                                                        action_blocker_model_type=action_blocker_model_type,
                                                        action_blocker_timesteps=max_time_steps,
                                                        pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                        use_mse=use_mse)

                                                    new_row = {
                                                        'batch_size': batch_size,
                                                        'hidden_layer_size': hidden_layer_size,
                                                        'algorithm_type': algorithm_type,
                                                        'optimizer': optimizer_type.name.lower(),
                                                        'learning_rate': learning_rate,
                                                        'goal_focused': 'Yes' if goal else 'No',
                                                        'is_double': 'Yes' if is_double else 'No',
                                                        'assign_priority': 'Yes' if assign_priority else 'No',
                                                        'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                                        'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                                        'action_blocker_model_type': action_blocker_model_type if enable_action_blocker else 'N/A',
                                                        'use_mse': 'Yes' if use_mse else 'No'
                                                    }

                                                    result = run_gym_env(env, agent, n_games_train=500, n_games_test=50)

                                                    for key in result:
                                                        new_row.update({key: result[key]})

                                                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_thompson_sampling(env, env_name, penalty, env_goal=None):
    assert penalty > 0
    pre_loaded_memory = develop_memory_from_gym_env(env, max_time_steps)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_td_thompson_sampling.csv'.format(env_name))
    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate', 'goal_focused',
                   'assign_priority', 'is_double', 'algorithm_type',
                   'enable_action_blocker', 'action_blocker_model_type', 'use_preloaded_memory', 'use_mse',
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
                                for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                               64, 128, 256, 512}):
                                    for goal in list({None, env_goal}):
                                        for assign_priority in [False, True]:
                                            for use_preloaded_memory in [False, True]:
                                                for use_mse in [False, True]:
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
                                                        goal=goal,
                                                        is_double=is_double,
                                                        algorithm_type=algorithm_type,
                                                        policy_type=PolicyType.THOMPSON_SAMPLING,
                                                        policy_args=policy_args,
                                                        assign_priority=assign_priority,
                                                        action_blocker_model_type=action_blocker_model_type,
                                                        action_blocker_timesteps=max_time_steps,
                                                        pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                        use_mse=use_mse)

                                                    result = run_gym_env(env, agent, n_games_train=500, n_games_test=50)

                                                    new_row = {
                                                        'batch_size': batch_size,
                                                        'hidden_layer_size': hidden_layer_size,
                                                        'algorithm_type': algorithm_type,
                                                        'optimizer': optimizer_type.name.lower(),
                                                        'learning_rate': learning_rate,
                                                        'goal_focused': 'Yes' if goal else 'No',
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


def run_all_td_methods(env, env_name, penalty, env_goal=None):
    print('Running', env_name)
    run_td_epsilon_greedy(env, env_name, penalty, env_goal)
    run_dueling_td_epsilon_greedy(env, env_name, penalty, env_goal)
    run_td_softmax(env, env_name, penalty, env_goal)
    run_dueling_td_softmax(env, env_name, penalty, env_goal)
    run_td_ucb(env, env_name, penalty, env_goal)
    run_dueling_td_ucb(env, env_name, penalty, env_goal)
    if penalty > 0:
        run_td_thompson_sampling(env, env_name, penalty, env_goal)
        run_dueling_td_thompson_sampling(env, env_name, penalty, env_goal)


def run_hill_climbing(env, env_name, penalty):
    pre_loaded_memory = develop_memory_from_gym_env(env, max_time_steps)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_hill_climbing.csv'.format(env_name))
    result_cols = ['enable_action_blocker', 'use_preloaded_memory',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for enable_action_blocker in list({False, penalty > 0}):
        action_blocker_model_types = ['decision_tree', 'random_forest'] if enable_action_blocker else [None]
        for action_blocker_model_type in action_blocker_model_types:
            for use_preloaded_memory in [False, True]:
                agent = HillClimbingAgent(input_dims=env.observation_space.shape, action_space=env.action_space,
                                          enable_action_blocking=enable_action_blocker,
                                          action_blocker_model_type=action_blocker_model_type,
                                          action_blocker_memory=pre_loaded_memory if use_preloaded_memory else None,
                                          action_blocker_timesteps=max_time_steps,
                                          gamma=1.0)

                new_row = {'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                           }

                result = run_gym_env(env, agent, n_games_train=500, n_games_test=50)

                for key in result:
                    new_row.update({key: result[key]})

                results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_ddpg(env, env_name, env_goal):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_ddpg.csv'.format(env_name))

    pre_loaded_memory = develop_memory_from_gym_env(env, max_time_steps)

    result_cols = ['batch_size', 'hidden_layer_size', 'actor_learning_rate',
                   'critic_learning_rate', 'tau', 'goal_focused', 'assign_priority', 'use_preloaded_memory', 'use_mse',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for batch_size in [64, 128]:
        for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                       64, 128, 256, 512}):
            for actor_learning_rate in [0.001, 0.0001]:
                for critic_learning_rate in [0.001, 0.0001]:
                    for tau in [1e-2, 1e-3]:
                        for goal in list({None, env_goal}):
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
                                        agent = DDPGAgent(
                                            input_dims=env.observation_space.shape,
                                            action_space=env.action_space,
                                            tau=tau, network_args=network_args,
                                            batch_size=batch_size,
                                            actor_optimizer_type=NetworkOptimizer.ADAM,
                                            critic_optimizer_type=NetworkOptimizer.ADAM,
                                            actor_optimizer_args=actor_optimizer_args,
                                            critic_optimizer_args=critic_optimizer_args,
                                            goal=goal,
                                            assign_priority=assign_priority,
                                            pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                            use_mse=use_mse
                                        )

                                        result = run_gym_env(env, agent, n_games_train=500, n_games_test=50)

                                        new_row = {
                                            'batch_size': batch_size,
                                            'hidden_layer_size': hidden_layer_size,
                                            'actor_learning_rate': actor_learning_rate,
                                            'critic_learning_rate': critic_learning_rate,
                                            'tau': tau,
                                            'goal_focused': 'Yes' if goal else 'No',
                                            'assign_priority': 'Yes' if assign_priority else 'No',
                                            'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                            'use_mse': 'Yes' if use_mse else 'No'
                                        }

                                        for key in result:
                                            new_row.update({key: result[key]})

                                        results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td3(env, env_name, env_goal):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_td3.csv'.format(env_name))

    pre_loaded_memory = develop_memory_from_gym_env(env, max_time_steps)

    result_cols = ['batch_size', 'hidden_layer_size', 'actor_learning_rate',
                   'critic_learning_rate', 'tau', 'goal_focused', 'assign_priority', 'use_preloaded_memory', 'use_mse',
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
                for goal in list({None, env_goal}):
                    for assign_priority in [False, True]:
                        for use_preloaded_memory in [False, True]:
                            for use_mse in [False, True]:
                                network_args = {
                                    'fc_dims': hidden_layer_size
                                }
                                agent = TD3Agent(
                                    input_dims=env.observation_space.shape,
                                    action_space=env.action_space,
                                    tau=tau, network_args=network_args,
                                    batch_size=batch_size,
                                    actor_optimizer_type=NetworkOptimizer.ADAM,
                                    critic_optimizer_type=NetworkOptimizer.ADAM,
                                    actor_optimizer_args=actor_optimizer_args,
                                    critic_optimizer_args=critic_optimizer_args,
                                    goal=goal,
                                    assign_priority=assign_priority,
                                    pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                    use_mse=use_mse
                                )

                                result = run_gym_env(env, agent, n_games_train=500, n_games_test=50)

                                new_row = {
                                    'batch_size': batch_size,
                                    'hidden_layer_size': hidden_layer_size,
                                    'tau': tau,
                                    'goal_focused': 'Yes' if goal else 'No',
                                    'assign_priority': 'Yes' if assign_priority else 'No',
                                    'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                    'use_mse': 'Yes' if use_mse else 'No'
                                }

                                for key in result:
                                    new_row.update({key: result[key]})

                                results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_actor_critic_continuous_methods(env, env_name, env_goal=None):
    print('Running', env_name)
    run_ddpg(env, env_name, env_goal)
    run_td3(env, env_name, env_goal)


def run_cem(env, env_name, env_goal=None):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_cem.csv'.format(env_name))

    result_cols = ['hidden_layer_size', 'goal_focused',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for hidden_layer_size in list({64, 128, 256, 512}):
        for goal in list({None, env_goal}):
            network_args = {
                'fc_dim': hidden_layer_size
            }
            agent = CEMAgent(input_dims=env.observation_space.shape, action_space=env.action_shape,
                             goal=goal, network_args=network_args)

            new_row = {'hidden_layer_size': hidden_layer_size, 'goal_focused': 'Yes' if goal else 'No'}

            result = run_gym_env(env, agent, n_games_train=500, n_games_test=50)

            for key in result:
                new_row.update({key: result[key]})

            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, float_format='%.3f', index=False)


def run_decision_tree_heuristics(env, env_name, heuristic_func, min_penalty=0, **args):
    pre_loaded_memory = develop_memory_from_gym_env(env, max_time_steps)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_dt.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'enable_action_blocking',
                   'use_preloaded_memory', 'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for learning_type in [LearningType.OFFLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for enable_action_blocking in list({False, min_penalty > 0}):
                for use_preloaded_memory in [False, True]:
                    agent = HeuristicWithDT(env.observation_space.shape, heuristic_func, use_model_only,
                                            env.action_space,
                                            enable_action_blocking, min_penalty,
                                            pre_loaded_memory if use_preloaded_memory else None,
                                            None, None, max_time_steps, 'decision_tree', **args)

                    result = run_gym_env(env, agent, learning_type=learning_type, n_games_train=500, n_games_test=50)

                    new_row = {
                        'learning_type': learning_type.name,
                        'use_model_only': 'Yes' if use_model_only else 'No',
                        'enable_action_blocking': 'Yes' if enable_action_blocking else 'No',
                        'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No'
                    }

                    for key in result:
                        new_row.update({key: result[key]})

                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, float_format='%.3f', index=False)


def run_random_forest_heuristics(env, env_name, heuristic_func, min_penalty=0, **args):
    pre_loaded_memory = develop_memory_from_gym_env(env, max_time_steps)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_rf.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'enable_action_blocking',
                   'use_preloaded_memory', 'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for learning_type in [LearningType.OFFLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for enable_action_blocking in list({False, min_penalty > 0}):
                for use_preloaded_memory in [False, True]:
                    agent = HeuristicWithRF(env.observation_space.shape, heuristic_func, use_model_only,
                                            env.action_space,
                                            enable_action_blocking, min_penalty,
                                            pre_loaded_memory if use_preloaded_memory else None,
                                            None, None, max_time_steps, 'random_forest', **args)

                    result = run_gym_env(env, agent, learning_type=learning_type, n_games_train=500, n_games_test=50)

                    new_row = {
                        'learning_type': learning_type.name,
                        'use_model_only': 'Yes' if use_model_only else 'No',
                        'enable_action_blocking': 'Yes' if enable_action_blocking else 'No',
                        'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No'
                    }

                    for key in result:
                        new_row.update({key: result[key]})

                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, float_format='%.3f', index=False)


def run_td_epsilon_greedy_heuristics(env, env_name, heuristic_func, penalty, env_goal=None, **args):
    pre_loaded_memory = develop_memory_from_gym_env(env, max_time_steps)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_td_epsilon_greedy.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size',
                   'optimizer', 'learning_rate', 'goal_focused', 'is_double', 'algorithm_type', 'enable_decay',
                   'epsilon_start', 'add_conservative_loss', 'alpha', 'enable_action_blocker',
                   'action_blocker_model_type', 'use_preloaded_memory', 'use_mse',
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
                                        for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                                       64, 128, 256, 512}):
                                            for goal in list({None, env_goal}):
                                                for add_conservative_loss in [False, True]:
                                                    for alpha in [0.001]:
                                                        for enable_decay in [False, True]:
                                                            epsilons = [1.0] if enable_decay \
                                                                else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                                                            for epsilon in epsilons:
                                                                for use_preloaded_memory in [False, True]:
                                                                    for use_mse in [False, True]:
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
                                                                            goal=goal,
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

                                                                        result = run_gym_env(env, agent,
                                                                                             learning_type=learning_type,
                                                                                             n_games_train=500,
                                                                                             n_games_test=50)

                                                                        new_row = {
                                                                            'batch_size': batch_size,
                                                                            'hidden_layer_size': hidden_layer_size,
                                                                            'algorithm_type': algorithm_type,
                                                                            'optimizer': optimizer_type.name.lower(),
                                                                            'learning_rate': learning_rate,
                                                                            'goal_focused': 'Yes' if goal else 'No',
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

                                                                        results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td_softmax_heuristics(env, env_name, heuristic_func, penalty, env_goal=None, **args):
    pre_loaded_memory = develop_memory_from_gym_env(env, max_time_steps)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_td_softmax.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size',
                   'optimizer', 'learning_rate', 'goal_focused', 'is_double', 'algorithm_type', 'tau',
                   'add_conservative_loss', 'alpha', 'enable_action_blocker', 'action_blocker_model_type',
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
                                        for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                                       64, 128, 256, 512}):
                                            for goal in list({None, env_goal}):
                                                for add_conservative_loss in [False, True]:
                                                    for alpha in [0.001]:
                                                        for tau in [0.0001, 0.001, 0.1, 1.0, 10.0]:
                                                            for use_preloaded_memory in [False, True]:
                                                                for use_mse in [False, True]:
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
                                                                        goal=goal,
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

                                                                    result = run_gym_env(env, agent,
                                                                                         learning_type=learning_type,
                                                                                         n_games_train=500, n_games_test=50)

                                                                    new_row = {
                                                                        'batch_size': batch_size,
                                                                        'hidden_layer_size': hidden_layer_size,
                                                                        'algorithm_type': algorithm_type,
                                                                        'optimizer': optimizer_type.name.lower(),
                                                                        'learning_rate': learning_rate,
                                                                        'goal_focused': 'Yes' if goal else 'No',
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

                                                                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td_ucb_heuristics(env, env_name, heuristic_func, penalty, env_goal=None, **args):
    pre_loaded_memory = develop_memory_from_gym_env(env, max_time_steps)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_td_ucb.csv'.format(env_name))

    policy_args = {'confidence_factor': 2}

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size',
                   'optimizer', 'learning_rate', 'goal_focused',
                   'is_double', 'algorithm_type', 'add_conservative_loss', 'alpha',
                   'enable_action_blocker', 'action_blocker_model_type', 'use_preloaded_memory', 'use_mse',
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
                                        for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                                       64, 128, 256, 512}):
                                            for goal in list({None, env_goal}):
                                                for add_conservative_loss in [False, True]:
                                                    for alpha in [0.001]:
                                                        for use_preloaded_memory in [False, True]:
                                                            for use_mse in [False, True]:
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
                                                                    goal=goal,
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

                                                                result = run_gym_env(env, agent,
                                                                                     learning_type=learning_type,
                                                                                     n_games_train=500, n_games_test=50)

                                                                new_row = {
                                                                    'batch_size': batch_size,
                                                                    'hidden_layer_size': hidden_layer_size,
                                                                    'algorithm_type': algorithm_type,
                                                                    'optimizer': optimizer_type.name.lower(),
                                                                    'learning_rate': learning_rate,
                                                                    'goal_focused': 'Yes' if goal else 'No',
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


def run_td_thompson_sampling_heuristics(env, env_name, heuristic_func, penalty, env_goal=None, **args):
    assert penalty > 0
    pre_loaded_memory = develop_memory_from_gym_env(env, max_time_steps)

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
                                        for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                                       64, 128, 256, 512}):
                                            for goal in list({None, env_goal}):
                                                for add_conservative_loss in [False, True]:
                                                    for alpha in [0.001]:
                                                        for use_preloaded_memory in [False, True]:
                                                            for use_mse in [False, True]:
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
                                                                    goal=goal,
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

                                                                result = run_gym_env(env, agent,
                                                                                     learning_type=learning_type,
                                                                                     n_games_train=500, n_games_test=50)

                                                                new_row = {
                                                                    'batch_size': batch_size,
                                                                    'hidden_layer_size': hidden_layer_size,
                                                                    'algorithm_type': algorithm_type,
                                                                    'optimizer': optimizer_type.name.lower(),
                                                                    'learning_rate': learning_rate,
                                                                    'goal_focused': 'Yes' if goal else 'No',
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


def run_dueling_td_epsilon_greedy_heuristics(env, env_name, heuristic_func, penalty, env_goal=None, **args):
    pre_loaded_memory = develop_memory_from_gym_env(env, max_time_steps)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_dueling_td_epsilon_greedy.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size',
                   'optimizer', 'learning_rate', 'goal_focused',
                   'is_double', 'algorithm_type', 'enable_decay', 'epsilon_start',
                   'add_conservative_loss', 'alpha', 'enable_action_blocker', 'action_blocker_model_type',
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
                                        for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                                       64, 128, 256, 512}):
                                            for goal in list({None, env_goal}):
                                                for add_conservative_loss in [False, True]:
                                                    for alpha in [0.001]:
                                                        for enable_decay in [False, True]:
                                                            epsilons = [1.0] if enable_decay \
                                                                else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                                                            for epsilon in epsilons:
                                                                for use_preloaded_memory in [False, True]:
                                                                    for use_mse in [False, True]:
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
                                                                            goal=goal,
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

                                                                        result = run_gym_env(env, agent,
                                                                                             learning_type=learning_type,
                                                                                             n_games_train=500,
                                                                                             n_games_test=50)

                                                                        new_row = {
                                                                            'batch_size': batch_size,
                                                                            'hidden_layer_size': hidden_layer_size,
                                                                            'algorithm_type': algorithm_type,
                                                                            'optimizer': optimizer_type.name.lower(),
                                                                            'learning_rate': learning_rate,
                                                                            'goal_focused': 'Yes' if goal else 'No',
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

                                                                        results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_softmax_heuristics(env, env_name, heuristic_func, penalty, env_goal=None, **args):
    pre_loaded_memory = develop_memory_from_gym_env(env, max_time_steps)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_dueling_td_softmax.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size',
                   'optimizer', 'learning_rate', 'goal_focused', 'is_double', 'algorithm_type', 'tau',
                   'add_conservative_loss', 'alpha', 'enable_action_blocker', 'action_blocker_model_type',
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
                                        for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                                       64, 128, 256, 512}):
                                            for goal in list({None, env_goal}):
                                                for add_conservative_loss in [False, True]:
                                                    for alpha in [0.001]:
                                                        for tau in [0.0001, 0.001, 0.1, 1.0, 10.0]:
                                                            for use_preloaded_memory in [False, True]:
                                                                for use_mse in [False, True]:
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
                                                                        goal=goal,
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

                                                                    result = run_gym_env(env, agent,
                                                                                         learning_type=learning_type,
                                                                                         n_games_train=500, n_games_test=50)

                                                                    new_row = {
                                                                        'batch_size': batch_size,
                                                                        'hidden_layer_size': hidden_layer_size,
                                                                        'algorithm_type': algorithm_type,
                                                                        'optimizer': optimizer_type.name.lower(),
                                                                        'learning_rate': learning_rate,
                                                                        'goal_focused': 'Yes' if goal else 'No',
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

                                                                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_ucb_heuristics(env, env_name, heuristic_func, penalty, env_goal=None, **args):
    pre_loaded_memory = develop_memory_from_gym_env(env, max_time_steps)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_dueling_td_ucb.csv'.format(env_name))

    policy_args = {'confidence_factor': 2}

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size',
                   'optimizer', 'learning_rate', 'goal_focused',
                   'is_double', 'algorithm_type', 'add_conservative_loss', 'alpha',
                   'enable_action_blocker', 'action_blocker_model_type', 'use_preloaded_memory', 'use_mse',
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
                                        for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                                       64, 128, 256, 512}):
                                            for goal in list({None, env_goal}):
                                                for add_conservative_loss in [False, True]:
                                                    for alpha in [0.001]:
                                                        for use_preloaded_memory in [False, True]:
                                                            for use_mse in [False, True]:
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
                                                                    goal=goal,
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

                                                                result = run_gym_env(env, agent,
                                                                                     learning_type=learning_type,
                                                                                     n_games_train=500, n_games_test=50)

                                                                new_row = {
                                                                    'batch_size': batch_size,
                                                                    'hidden_layer_size': hidden_layer_size,
                                                                    'algorithm_type': algorithm_type,
                                                                    'optimizer': optimizer_type.name.lower(),
                                                                    'learning_rate': learning_rate,
                                                                    'goal_focused': 'Yes' if goal else 'No',
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


def run_dueling_td_thompson_sampling_heuristics(env, env_name, heuristic_func, penalty, env_goal=None, **args):
    assert penalty > 0
    pre_loaded_memory = develop_memory_from_gym_env(env, max_time_steps)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_dueling_td_thompson_sampling.csv'.format(env_name))

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
                                        for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                                       64, 128, 256, 512}):
                                            for goal in list({None, env_goal}):
                                                for add_conservative_loss in [False, True]:
                                                    for alpha in [0.001]:
                                                        for use_preloaded_memory in [False, True]:
                                                            for use_mse in [False, True]:
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
                                                                    goal=goal,
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

                                                                result = run_gym_env(env, agent,
                                                                                     learning_type=learning_type,
                                                                                     n_games_train=500, n_games_test=50)

                                                                new_row = {
                                                                    'batch_size': batch_size,
                                                                    'hidden_layer_size': hidden_layer_size,
                                                                    'algorithm_type': algorithm_type,
                                                                    'optimizer': optimizer_type.name.lower(),
                                                                    'learning_rate': learning_rate,
                                                                    'goal_focused': 'Yes' if goal else 'No',
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
    pre_loaded_memory = develop_memory_from_gym_env(env, max_time_steps)

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
                        agent = HeuristicWithHillClimbing(input_dims=env.observation_space.shape,
                                                          action_space=env.action_space,
                                                          enable_action_blocking=enable_action_blocker,
                                                          action_blocker_model_type=action_blocker_model_type,
                                                          action_blocker_timesteps=max_time_steps,
                                                          pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                          gamma=1.0,
                                                          heuristic_func=heuristic_func,
                                                          use_model_only=use_model_only, **args)

                        new_row = {'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                   'use_model_only': 'Yes' if use_model_only else 'No',
                                   'learning_type': learning_type.name,
                                   'use_preloaded_memory': 'Yes' if pre_loaded_memory else 'No',
                                   'action_blocker_model_type': action_blocker_model_type if enable_action_blocker else 'N/A'
                                   }

                        result = run_gym_env(env, agent, n_games_train=500, n_games_test=50,
                                             learning_type=learning_type)

                        for key in result:
                            new_row.update({key: result[key]})

                        results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_ddpg_heuristics(env, env_name, heuristic_func, env_goal, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristics_ddpg.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size', 'actor_learning_rate',
                   'critic_learning_rate', 'tau', 'goal_focused', 'use_preloaded_memory', 'use_mse',
                   'num_time_steps_test', 'avg_score_test']

    pre_loaded_memory = develop_memory_from_gym_env(env, max_time_steps)

    results = pd.DataFrame(columns=result_cols)

    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for batch_size in [64, 128]:
                for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                               64, 128, 256, 512}):
                    for actor_learning_rate in [0.001, 0.0001]:
                        for critic_learning_rate in [0.001, 0.0001]:
                            for tau in [1e-2, 1e-3]:
                                for goal in list({None, env_goal}):
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
                                            agent = HeuristicWithDDPG(
                                                input_dims=env.observation_space.shape,
                                                action_space=env.action_space,
                                                tau=tau, network_args=network_args,
                                                batch_size=batch_size,
                                                actor_optimizer_type=NetworkOptimizer.ADAM,
                                                critic_optimizer_type=NetworkOptimizer.ADAM,
                                                actor_optimizer_args=actor_optimizer_args,
                                                critic_optimizer_args=critic_optimizer_args,
                                                goal=goal,
                                                learning_type=learning_type,
                                                use_model_only=use_model_only,
                                                pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                heuristic_func=heuristic_func, use_mse=use_mse, **args
                                            )

                                            result = run_gym_env(env, agent, learning_type=learning_type,
                                                                 n_games_train=500, n_games_test=50)

                                            new_row = {
                                                'learning_type': learning_type.name,
                                                'use_model_only': 'Yes' if use_model_only else 'No',
                                                'batch_size': batch_size,
                                                'hidden_layer_size': hidden_layer_size,
                                                'actor_learning_rate': actor_learning_rate,
                                                'critic_learning_rate': critic_learning_rate,
                                                'tau': tau,
                                                'goal_focused': 'Yes' if goal else 'No',
                                                'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                                'use_mse': 'Yes' if use_mse else 'No'
                                            }

                                            for key in result:
                                                new_row.update({key: result[key]})

                                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td3_heuristics(env, env_name, heuristic_func, env_goal, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristics_td3.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size', 'actor_learning_rate',
                   'critic_learning_rate', 'tau', 'goal_focused', 'use_preloaded_memory', 'use_mse',
                   'num_time_steps_test', 'avg_score_test']

    pre_loaded_memory = develop_memory_from_gym_env(env, max_time_steps)

    results = pd.DataFrame(columns=result_cols)

    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for batch_size in [64, 128]:
                for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                               64, 128, 256, 512}):
                    for actor_learning_rate in [0.001, 0.0001]:
                        for critic_learning_rate in [0.001, 0.0001]:
                            for tau in [1e-2, 1e-3]:
                                for goal in list({None, env_goal}):
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
                                            agent = HeuristicWithTD3(
                                                input_dims=env.observation_space.shape,
                                                action_space=env.action_space,
                                                tau=tau, network_args=network_args,
                                                batch_size=batch_size,
                                                actor_optimizer_type=NetworkOptimizer.ADAM,
                                                critic_optimizer_type=NetworkOptimizer.ADAM,
                                                actor_optimizer_args=actor_optimizer_args,
                                                critic_optimizer_args=critic_optimizer_args,
                                                goal=goal,
                                                learning_type=learning_type,
                                                use_model_only=use_model_only,
                                                pre_loaded_memory=pre_loaded_memory if use_preloaded_memory else None,
                                                heuristic_func=heuristic_func, use_mse=use_mse, **args
                                            )

                                            result = run_gym_env(env, agent, learning_type=learning_type,
                                                                 n_games_train=500, n_games_test=50)

                                            new_row = {
                                                'learning_type': learning_type.name,
                                                'use_model_only': 'Yes' if use_model_only else 'No',
                                                'batch_size': batch_size,
                                                'hidden_layer_size': hidden_layer_size,
                                                'actor_learning_rate': actor_learning_rate,
                                                'critic_learning_rate': critic_learning_rate,
                                                'tau': tau,
                                                'goal_focused': 'Yes' if goal else 'No',
                                                'use_preloaded_memory': 'Yes' if use_preloaded_memory else 'No',
                                                'use_mse': 'Yes' if use_mse else 'No'
                                            }

                                            for key in result:
                                                new_row.update({key: result[key]})

                                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_cem_heuristics(env, env_name, heuristic_func, env_goal=None, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_cem.csv'.format(env_name))

    result_cols = ['use_model_only', 'learning_type', 'hidden_layer_size', 'goal_focused',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for use_model_only in [False, True]:
        for learning_type in [LearningType.OFFLINE, LearningType.OFFLINE, LearningType.BOTH]:
            for hidden_layer_size in list({64, 128, 256, 512}):
                for goal in list({None, env_goal}):
                    network_args = {
                        'fc_dim': hidden_layer_size
                    }
                    agent = HeuristicWithCEM(input_dims=env.observation_space.shape, action_space=env.action_shape,
                                             goal=goal, network_args=network_args, use_model_only=use_model_only,
                                             learning_type=learning_type, heuristic_func=heuristic_func, **args)

                    new_row = {'hidden_layer_size': hidden_layer_size, 'goal_focused': 'Yes' if goal else 'No',
                               'use_model_only': 'Yes' if use_model_only else 'No',
                               'learning_type': learning_type.name}

                    result = run_gym_env(env, agent, n_games_train=500, n_games_test=50, learning_type=learning_type)

                    for key in result:
                        new_row.update({key: result[key]})

                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, float_format='%.3f', index=False)


def run_heuristics(env, env_name, heuristic_func, penalty, **args):
    run_decision_tree_heuristics(env, env_name, heuristic_func, penalty, **args)
    run_random_forest_heuristics(env, env_name, heuristic_func, penalty, **args)
    run_td_epsilon_greedy_heuristics(env, env_name, heuristic_func, penalty,  **args)
    run_dueling_td_epsilon_greedy_heuristics(env, env_name, heuristic_func, penalty, **args)
    run_td_softmax_heuristics(env, env_name, heuristic_func, penalty, **args)
    run_dueling_td_softmax_heuristics(env, env_name, heuristic_func, penalty, **args)
    run_td_ucb_heuristics(env, env_name, heuristic_func, penalty, **args)
    run_dueling_td_ucb_heuristics(env, env_name, heuristic_func, penalty, **args)
    run_td_thompson_sampling_heuristics(env, env_name, heuristic_func, penalty, **args)
    run_dueling_td_thompson_sampling_heuristics(env, env_name, heuristic_func, penalty, **args)
    run_hill_climbing_heuristics(env, env_name, penalty, heuristic_func, **args)
