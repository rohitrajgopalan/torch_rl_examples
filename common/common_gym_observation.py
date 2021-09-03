import os

import pandas as pd
from gym.spaces import Discrete
from torch_rl.heuristic.heuristic_with_dt import HeuristicWithDT
from torch_rl.heuristic.heuristic_with_td import HeuristicWithTD
from torch_rl.heuristic.heuristic_with_td3 import HeuristicWithTD3
from torch_rl.heuristic.heuristic_with_dueling_td import HeuristicWithDuelingTD
from torch_rl.heuristic.heuristic_with_ddpg import HeuristicWithDDPG
from torch_rl.td.agent import TDAgent
from torch_rl.dueling_td.agent import DuelingTDAgent
from torch_rl.ddpg.agent import DDPGAgent
from torch_rl.td3.agent import TD3Agent
from torch_rl.utils.types import NetworkOptimizer, TDAlgorithmType, PolicyType, LearningType

try:
    from run import run_gym_env
    from utils import develop_memory_for_dt_action_blocker_for_gym_env, derive_hidden_layer_size
except ImportError:
    from run import run_gym_env
    from utils import develop_memory_for_dt_action_blocker_for_gym_env, derive_hidden_layer_size


def run_td_epsilon_greedy(env, env_name, penalty, env_goal=None):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_td_epsilon_greedy.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate', 'goal_focused',
                   'assign_priority', 'enable_action_blocker', 'use_ml_for_action_blocker',
                   'is_double', 'algorithm_type', 'enable_decay', 'epsilon_start',
                   'num_time_steps_train', 'avg_score_train',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    action_blocker_memory = develop_memory_for_dt_action_blocker_for_gym_env(env)

    policy_args = {}

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in list({False, penalty > 0}):
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
                                                use_ml_flags = [False, True] if enable_action_blocker else [False]
                                                for use_ml_for_action_blocker in use_ml_flags:
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
                                                        use_ml_for_action_blocker=use_ml_for_action_blocker,
                                                        action_blocker_memory=action_blocker_memory)

                                                    result = run_gym_env(env, agent, n_games_train=500, n_games_test=50)

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
                                                        'use_ml_for_action_blocker': 'Yes' if use_ml_for_action_blocker else 'No'
                                                    }
                                                    for key in result:
                                                        new_row.update({key: result[key]})

                                                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td_softmax(env, env_name, penalty, env_goal=None):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_td_softmax.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate', 'goal_focused',
                   'assign_priority', 'enable_action_blocker', 'use_ml_for_action_blocker',
                   'is_double', 'algorithm_type', 'tau', 'num_time_steps_train', 'avg_score_train',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    action_blocker_memory = develop_memory_for_dt_action_blocker_for_gym_env(env)
    policy_args = {}

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in list({False, penalty > 0}):
                for batch_size in [32, 64, 128]:
                    for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                        for learning_rate in [0.001, 0.0001]:
                            for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                           64, 128, 256, 512}):
                                for goal in list({None, env_goal}):
                                    for tau in [0.0001, 0.001, 0.1, 1.0, 10.0]:
                                        policy_args.update({'tau': tau})
                                        for assign_priority in [False, True]:
                                            use_ml_flags = [False, True] if enable_action_blocker else [False]
                                            for use_ml_for_action_blocker in use_ml_flags:
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
                                                    use_ml_for_action_blocker=use_ml_for_action_blocker,
                                                    action_blocker_memory=action_blocker_memory)

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
                                                    'use_ml_for_action_blocker': 'Yes' if use_ml_for_action_blocker else 'No'
                                                }
                                                for key in result:
                                                    new_row.update({key: result[key]})

                                                results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td_ucb(env, env_name, penalty, env_goal=None):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results', '{0}_td_ucb.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate', 'goal_focused',
                   'assign_priority', 'is_double', 'algorithm_type',
                   'enable_action_blocker', 'use_ml_for_action_blocker',
                   'num_time_steps_train', 'avg_score_train',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    action_blocker_memory = develop_memory_for_dt_action_blocker_for_gym_env(env)

    policy_args = {'confidence_factor': 2}

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in list({False, penalty > 0}):
                for batch_size in [32, 64, 128]:
                    for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                        for learning_rate in [0.001, 0.0001]:
                            for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                           64, 128, 256, 512}):
                                for goal in list({None, env_goal}):
                                    for assign_priority in [False, True]:
                                        use_ml_flags = [False, True] if enable_action_blocker else [False]
                                        for use_ml_for_action_blocker in use_ml_flags:
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
                                                use_ml_for_action_blocker=use_ml_for_action_blocker,
                                                action_blocker_memory=action_blocker_memory)

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
                                                'use_ml_for_action_blocker': 'Yes' if use_ml_for_action_blocker else 'No'
                                            }
                                            for key in result:
                                                new_row.update({key: result[key]})

                                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td_thompson_sampling(env, env_name, penalty, env_goal=None):
    assert penalty > 0
    action_blocker_memory = develop_memory_for_dt_action_blocker_for_gym_env(env)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_td_thompson_sampling.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate', 'goal_focused',
                   'assign_priority', 'is_double', 'algorithm_type',
                   'enable_action_blocker', 'use_ml_for_action_blocker',
                   'num_time_steps_train', 'avg_score_train',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {
        'min_penalty': penalty
    }

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in list({False, penalty > 0}):
                for batch_size in [32, 64, 128]:
                    for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                        for learning_rate in [0.001, 0.0001]:

                            for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                           64, 128, 256, 512}):
                                for goal in list({None, env_goal}):
                                    for assign_priority in [False, True]:
                                        use_ml_flags = [False, True] if enable_action_blocker else [False]
                                        for use_ml_for_action_blocker in use_ml_flags:
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
                                                use_ml_for_action_blocker=use_ml_for_action_blocker,
                                                action_blocker_memory=action_blocker_memory)

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
                                                'use_ml_for_action_blocker': 'Yes' if use_ml_for_action_blocker else 'No'
                                            }
                                            for key in result:
                                                new_row.update({key: result[key]})

                                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_epsilon_greedy(env, env_name, penalty, env_goal=None):
    action_blocker_memory = develop_memory_for_dt_action_blocker_for_gym_env(env)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_td_epsilon_greedy.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate', 'goal_focused',
                   'using_move_matrix', 'is_double', 'algorithm_type', 'enable_decay',
                   'epsilon_start', 'enable_action_blocker', 'use_ml_for_action_blocker',
                   'num_time_steps_train', 'avg_score_train',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in list({False, penalty > 0}):
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
                                                for use_ml_for_action_blocker in list(
                                                        {False, enable_action_blocker}):
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
                                                        use_ml_for_action_blocker=use_ml_for_action_blocker,
                                                        action_blocker_memory=action_blocker_memory)

                                                    result = run_gym_env(env, agent, n_games_train=500, n_games_test=50)

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
                                                        'use_ml_for_action_blocker': 'Yes' if use_ml_for_action_blocker else 'No'
                                                    }
                                                    for key in result:
                                                        new_row.update({key: result[key]})

                                                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_softmax(env, env_name, penalty, env_goal=None):
    action_blocker_memory = develop_memory_for_dt_action_blocker_for_gym_env(env)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_td_softmax.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate', 'goal_focused',
                   'assign_priority', 'is_double', 'algorithm_type', 'tau',
                   'enable_action_blocker', 'use_ml_for_action_blocker',
                   'num_time_steps_train', 'avg_score_train',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in list({False, penalty > 0}):
                for batch_size in [32, 64, 128]:
                    for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                        for learning_rate in [0.001, 0.0001]:

                            for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                           64, 128, 256, 512}):
                                for goal in list({None, env_goal}):
                                    for tau in [0.0001, 0.001, 0.1, 1.0, 10.0]:
                                        policy_args.update({'tau': tau})
                                        for assign_priority in [False, True]:
                                            for use_ml_for_action_blocker in list(
                                                    {False, enable_action_blocker}):

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
                                                    use_ml_for_action_blocker=use_ml_for_action_blocker,
                                                    action_blocker_memory=action_blocker_memory)

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
                                                    'use_ml_for_action_blocker': 'Yes' if use_ml_for_action_blocker else 'No'
                                                }
                                                for key in result:
                                                    new_row.update({key: result[key]})

                                                results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_ucb(env, env_name, penalty, env_goal=None):
    action_blocker_memory = develop_memory_for_dt_action_blocker_for_gym_env(env)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_td_ucb.csv'.format(env_name))
    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate', 'goal_focused',
                   'assign_priority', 'is_double', 'algorithm_type',
                   'enable_action_blocker', 'use_ml_for_action_blocker',
                   'num_time_steps_train', 'avg_score_train',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {'confidence_factor': 2}

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in list({False, penalty > 0}):
                for batch_size in [32, 64, 128]:
                    for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                        for learning_rate in [0.001, 0.0001]:

                            for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                           64, 128, 256, 512}):
                                for goal in list({None, env_goal}):
                                    for assign_priority in [False, True]:
                                        for use_ml_for_action_blocker in list({False, enable_action_blocker}):
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
                                                use_ml_for_action_blocker=use_ml_for_action_blocker,
                                                action_blocker_memory=action_blocker_memory)

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
                                                'use_ml_for_action_blocker': 'Yes' if use_ml_for_action_blocker else 'No'
                                            }

                                            result = run_gym_env(env, agent, n_games_train=500, n_games_test=50)

                                            for key in result:
                                                new_row.update({key: result[key]})

                                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_thompson_sampling(env, env_name, penalty, env_goal=None):
    assert penalty > 0
    action_blocker_memory = develop_memory_for_dt_action_blocker_for_gym_env(env)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_td_thompson_sampling.csv'.format(env_name))
    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate', 'goal_focused',
                   'assign_priority', 'is_double', 'algorithm_type',
                   'enable_action_blocker', 'use_ml_for_action_blocker',
                   'num_time_steps_train', 'avg_score_train',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {
        'min_penalty': penalty
    }

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in list({False, penalty > 0}):
                for batch_size in [32, 64, 128]:
                    for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                        for learning_rate in [0.001, 0.0001]:

                            for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                           64, 128, 256, 512}):
                                for goal in list({None, env_goal}):
                                    for assign_priority in [False, True]:
                                        for use_ml_for_action_blocker in list({False, enable_action_blocker}):

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
                                                use_ml_for_action_blocker=use_ml_for_action_blocker,
                                                action_blocker_memory=action_blocker_memory)

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
                                                'use_ml_for_action_blocker': 'Yes' if use_ml_for_action_blocker else 'No'
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


def run_ddpg(env, env_name, env_goal):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_ddpg.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'actor_learning_rate',
                   'critic_learning_rate', 'tau', 'goal_focused', 'assign_priority',
                   'num_time_steps_train', 'avg_score_train',
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
                                    assign_priority=assign_priority
                                )

                                result = run_gym_env(env, agent, n_games_train=500, n_games_test=50)

                                new_row = {
                                    'batch_size': batch_size,
                                    'hidden_layer_size': hidden_layer_size,
                                    'actor_learning_rate': actor_learning_rate,
                                    'critic_learning_rate': critic_learning_rate,
                                    'tau': tau,
                                    'goal_focused': 'Yes' if goal else 'No',
                                    'assign_priority': 'Yes' if assign_priority else 'No'
                                }

                                for key in result:
                                    new_row.update({key: result[key]})

                                results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td3(env, env_name, env_goal):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_td3.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'actor_learning_rate',
                   'critic_learning_rate', 'tau', 'goal_focused', 'assign_priority',
                   'num_time_steps_train', 'avg_score_train',
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
                            assign_priority=assign_priority
                        )

                        result = run_gym_env(env, agent, n_games_train=500, n_games_test=50)

                        new_row = {
                            'batch_size': batch_size,
                            'hidden_layer_size': hidden_layer_size,
                            'tau': tau,
                            'goal_focused': 'Yes' if goal else 'No',
                            'assign_priority': 'Yes' if assign_priority else 'No'
                        }

                        for key in result:
                            new_row.update({key: result[key]})

                        results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_actor_critic_continuous_methods(env, env_name, env_goal=None):
    print('Running', env_name)
    run_ddpg(env, env_name, env_goal)
    run_td3(env, env_name, env_goal)


def run_decision_tree_heuristics(env, env_name, heuristic_func, min_penalty=0, **args):
    action_blocker_memory = develop_memory_for_dt_action_blocker_for_gym_env(env)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_dt.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'enable_action_blocking', 'use_ml_for_action_blocker',
                   'num_time_steps_train', 'avg_score_train',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for learning_type in [LearningType.OFFLINE]:
        for use_model_only in [False, True]:
            for enable_action_blocking in list({False, min_penalty > 0}):
                for use_ml_for_action_blocker in list({False, enable_action_blocking}):

                    agent = HeuristicWithDT(heuristic_func, use_model_only, env.action_space,
                                            enable_action_blocking, min_penalty, use_ml_for_action_blocker,
                                            action_blocker_memory, None, None, **args)

                    result = run_gym_env(env, agent, learning_type=learning_type, n_games_train=500, n_games_test=50)

                    new_row = {
                        'learning_type': learning_type.name,
                        'use_model_only': 'Yes' if use_model_only else 'No',
                        'enable_action_blocking': 'Yes' if enable_action_blocking else 'No',
                        'use_ml_for_action_blocker': 'Yes' if use_ml_for_action_blocker else 'No',
                    }

                    for key in result:
                        new_row.update({key: result[key]})

                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, float_format='%.3f', index=False)


def run_td_epsilon_greedy_heuristics(env, env_name, heuristic_func, penalty, env_goal=None, **args):
    action_blocker_memory = develop_memory_for_dt_action_blocker_for_gym_env(env)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_td_epsilon_greedy.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size',
                   'optimizer', 'learning_rate', 'goal_focused', 'is_double', 'algorithm_type', 'enable_decay',
                   'epsilon_start', 'add_conservative_loss', 'alpha', 'enable_action_blocker',
                   'use_ml_for_action_blocker',
                   'num_time_steps_train', 'avg_score_train',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}
    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for is_double in [False, True]:
                for algorithm_type in TDAlgorithmType.all():
                    for enable_action_blocker in list({False, penalty > 0}):
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
                                                            for use_ml_for_action_blocker in list({False,
                                                                                                   enable_action_blocker}):
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
                                                                    use_ml_for_action_blocker=use_ml_for_action_blocker,
                                                                    action_blocker_memory=action_blocker_memory,
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
                                                                    'enable_decay': 'Yes' if enable_decay else 'No',
                                                                    'epsilon': epsilon,
                                                                    'learning_type': learning_type.name,
                                                                    'use_model_only': 'Yes' if use_model_only else 'No',
                                                                    'add_conservative_loss': 'Yes' if add_conservative_loss else 'No',
                                                                    'alpha': alpha if add_conservative_loss else 0,
                                                                    'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                                                    'use_ml_for_action_blocker': 'Yes' if use_ml_for_action_blocker else 'No',
                                                                }
                                                                for key in result:
                                                                    new_row.update({key: result[key]})

                                                                results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td_softmax_heuristics(env, env_name, heuristic_func, penalty, env_goal=None, **args):
    action_blocker_memory = develop_memory_for_dt_action_blocker_for_gym_env(env)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_td_softmax.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size',
                   'optimizer', 'learning_rate', 'goal_focused', 'is_double', 'algorithm_type', 'tau',
                   'add_conservative_loss', 'alpha', 'enable_action_blocker', 'use_ml_for_action_blocker',
                   'num_time_steps_train', 'avg_score_train',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}
    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for is_double in [False, True]:
                for algorithm_type in TDAlgorithmType.all():
                    for enable_action_blocker in list({False, penalty > 0}):
                        for batch_size in [32, 64, 128]:
                            for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                                for learning_rate in [0.001, 0.0001]:
                                    for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                                   64, 128, 256, 512}):
                                        for goal in list({None, env_goal}):
                                            for add_conservative_loss in [False, True]:
                                                for alpha in [0.001]:
                                                    for tau in [0.0001, 0.001, 0.1, 1.0, 10.0]:
                                                        for use_ml_for_action_blocker in list(
                                                                {False, enable_action_blocker}):
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
                                                                use_ml_for_action_blocker=use_ml_for_action_blocker,
                                                                action_blocker_memory=action_blocker_memory,
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
                                                                'use_ml_for_action_blocker': 'Yes' if use_ml_for_action_blocker else 'No',
                                                            }
                                                            for key in result:
                                                                new_row.update({key: result[key]})

                                                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td_ucb_heuristics(env, env_name, heuristic_func, penalty, env_goal=None, **args):
    action_blocker_memory = develop_memory_for_dt_action_blocker_for_gym_env(env)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_td_ucb.csv'.format(env_name))

    policy_args = {'confidence_factor': 2}

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size',
                   'optimizer', 'learning_rate', 'goal_focused',
                   'is_double', 'algorithm_type', 'add_conservative_loss', 'alpha',
                   'enable_action_blocker', 'use_ml_for_action_blocker',
                   'num_time_steps_train', 'avg_score_train',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for is_double in [False, True]:
                for algorithm_type in TDAlgorithmType.all():
                    for enable_action_blocker in list({False, penalty > 0}):
                        for batch_size in [32, 64, 128]:
                            for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                                for learning_rate in [0.001, 0.0001]:
                                    for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                                   64, 128, 256, 512}):
                                        for goal in list({None, env_goal}):
                                            for add_conservative_loss in [False, True]:
                                                for alpha in [0.001]:
                                                    for use_ml_for_action_blocker in list(
                                                            {False, enable_action_blocker}):
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
                                                            use_ml_for_action_blocker=use_ml_for_action_blocker,
                                                            action_blocker_memory=action_blocker_memory,
                                                            alpha=alpha, **args)

                                                        result = run_gym_env(env, agent, learning_type=learning_type,
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
                                                            'use_ml_for_action_blocker': 'Yes' if use_ml_for_action_blocker else 'No',
                                                        }
                                                        for key in result:
                                                            new_row.update({key: result[key]})

                                                        results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td_thompson_sampling_heuristics(env, env_name, heuristic_func, penalty, env_goal=None, **args):
    assert penalty > 0
    action_blocker_memory = develop_memory_for_dt_action_blocker_for_gym_env(env)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_td_thompson_sampling.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size',
                   'optimizer', 'learning_rate', 'goal_focused',
                   'is_double', 'algorithm_type', 'add_conservative_loss', 'alpha',
                   'enable_action_blocker', 'use_ml_for_action_blocker',
                   'num_time_steps_train', 'avg_score_train',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}
    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for is_double in [False, True]:
                for algorithm_type in TDAlgorithmType.all():
                    for enable_action_blocker in list({False, penalty > 0}):
                        for batch_size in [32, 64, 128]:
                            for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                                for learning_rate in [0.001, 0.0001]:
                                    for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                                   64, 128, 256, 512}):
                                        for goal in list({None, env_goal}):
                                            for add_conservative_loss in [False, True]:
                                                for alpha in [0.001]:
                                                    for use_ml_for_action_blocker in list(
                                                            {False, enable_action_blocker}):
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
                                                            use_ml_for_action_blocker=use_ml_for_action_blocker,
                                                            action_blocker_memory=action_blocker_memory,
                                                            alpha=alpha, **args)

                                                        result = run_gym_env(env, agent, learning_type=learning_type,
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
                                                            'use_ml_for_action_blocker': 'Yes' if use_ml_for_action_blocker else 'No',
                                                        }
                                                        for key in result:
                                                            new_row.update({key: result[key]})

                                                        results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_epsilon_greedy_heuristics(env, env_name, heuristic_func, penalty, env_goal=None, **args):
    action_blocker_memory = develop_memory_for_dt_action_blocker_for_gym_env(env)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_dueling_td_epsilon_greedy.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size',
                   'optimizer', 'learning_rate', 'goal_focused',
                   'is_double', 'algorithm_type', 'enable_decay', 'epsilon_start',
                   'add_conservative_loss', 'alpha', 'enable_action_blocker', 'use_ml_for_action_blocker',
                   'num_time_steps_train', 'avg_score_train',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}
    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for is_double in [False, True]:
                for algorithm_type in TDAlgorithmType.all():
                    for enable_action_blocker in list({False, penalty > 0}):
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
                                                            for use_ml_for_action_blocker in list(
                                                                    {False, enable_action_blocker}):
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
                                                                    use_ml_for_action_blocking=use_ml_for_action_blocker,
                                                                    action_blocker_memory=action_blocker_memory,
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
                                                                    'enable_decay': 'Yes' if enable_decay else 'No',
                                                                    'epsilon': epsilon,
                                                                    'learning_type': learning_type.name,
                                                                    'use_model_only': 'Yes' if use_model_only else 'No',
                                                                    'add_conservative_loss': 'Yes' if add_conservative_loss else 'No',
                                                                    'alpha': alpha if add_conservative_loss else 0,
                                                                    'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                                                    'use_ml_for_action_blocker': 'Yes' if use_ml_for_action_blocker else 'No',
                                                                }
                                                                for key in result:
                                                                    new_row.update({key: result[key]})

                                                                results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_softmax_heuristics(env, env_name, heuristic_func, penalty, env_goal=None, **args):
    action_blocker_memory = develop_memory_for_dt_action_blocker_for_gym_env(env)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_dueling_td_softmax.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size',
                   'optimizer', 'learning_rate', 'goal_focused', 'is_double', 'algorithm_type', 'tau',
                   'add_conservative_loss', 'alpha', 'enable_action_blocker', 'use_ml_for_action_blocker',
                   'num_time_steps_train', 'avg_score_train',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}
    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for is_double in [False, True]:
                for algorithm_type in TDAlgorithmType.all():
                    for enable_action_blocker in list({False, penalty > 0}):
                        for batch_size in [32, 64, 128]:
                            for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                                for learning_rate in [0.001, 0.0001]:
                                    for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                                   64, 128, 256, 512}):
                                        for goal in list({None, env_goal}):
                                            for add_conservative_loss in [False, True]:
                                                for alpha in [0.001]:
                                                    for tau in [0.0001, 0.001, 0.1, 1.0, 10.0]:
                                                        for use_ml_for_action_blocker in list(
                                                                {False, enable_action_blocker}):
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
                                                                use_ml_for_action_blocking=use_ml_for_action_blocker,
                                                                action_blocker_memory=action_blocker_memory,
                                                                use_model_only=use_model_only, **args)

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
                                                                'use_ml_for_action_blocker': 'Yes' if use_ml_for_action_blocker else 'No',
                                                            }
                                                            for key in result:
                                                                new_row.update({key: result[key]})

                                                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_ucb_heuristics(env, env_name, heuristic_func, penalty, env_goal=None, **args):
    action_blocker_memory = develop_memory_for_dt_action_blocker_for_gym_env(env)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_dueling_td_ucb.csv'.format(env_name))

    policy_args = {'confidence_factor': 2}

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size',
                   'optimizer', 'learning_rate', 'goal_focused',
                   'is_double', 'algorithm_type', 'add_conservative_loss', 'alpha',
                   'enable_action_blocker', 'use_ml_for_action_blocker',
                   'num_time_steps_train', 'avg_score_train',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for is_double in [False, True]:
                for algorithm_type in TDAlgorithmType.all():
                    for enable_action_blocker in list({False, penalty > 0}):
                        for batch_size in [32, 64, 128]:
                            for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                                for learning_rate in [0.001, 0.0001]:
                                    for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                                   64, 128, 256, 512}):
                                        for goal in list({None, env_goal}):
                                            for add_conservative_loss in [False, True]:
                                                for alpha in [0.001]:
                                                    for use_ml_for_action_blocker in list(
                                                            {False, enable_action_blocker}):
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
                                                            use_ml_for_action_blocking=use_ml_for_action_blocker,
                                                            action_blocker_memory=action_blocker_memory,
                                                            alpha=alpha, **args)

                                                        result = run_gym_env(env, agent, learning_type=learning_type,
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
                                                            'use_ml_for_action_blocker': 'Yes' if use_ml_for_action_blocker else 'No',
                                                        }
                                                        for key in result:
                                                            new_row.update({key: result[key]})

                                                        results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_thompson_sampling_heuristics(env, env_name, heuristic_func, penalty, env_goal=None, **args):
    assert penalty > 0
    action_blocker_memory = develop_memory_for_dt_action_blocker_for_gym_env(env)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristic_dueling_td_thompson_sampling.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size',
                   'optimizer', 'learning_rate', 'goal_focused',
                   'is_double', 'algorithm_type', 'add_conservative_loss', 'alpha',
                   'enable_action_blocker', 'use_ml_for_action_blocker',
                   'num_time_steps_train', 'avg_score_train',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}
    for learning_type in [LearningType.OFFLINE, LearningType.ONLINE, LearningType.BOTH]:
        for use_model_only in [False, True]:
            for is_double in [False, True]:
                for algorithm_type in TDAlgorithmType.all():
                    for enable_action_blocker in list({False, penalty > 0}):
                        for batch_size in [32, 64, 128]:
                            for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                                for learning_rate in [0.001, 0.0001]:
                                    for hidden_layer_size in list({derive_hidden_layer_size(env, batch_size),
                                                                   64, 128, 256, 512}):
                                        for goal in list({None, env_goal}):
                                            for add_conservative_loss in [False, True]:
                                                for alpha in [0.001]:
                                                    for use_ml_for_action_blocker in list(
                                                            {False, enable_action_blocker}):
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
                                                            use_ml_for_action_blocking=use_ml_for_action_blocker,
                                                            action_blocker_memory=action_blocker_memory,
                                                            alpha=alpha, **args)

                                                        result = run_gym_env(env, agent, learning_type=learning_type,
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
                                                            'use_ml_for_action_blocker': 'Yes' if use_ml_for_action_blocker else 'No',
                                                        }
                                                        for key in result:
                                                            new_row.update({key: result[key]})

                                                        results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_ddpg_heuristics(env, env_name, heuristic_func, env_goal, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristics_ddpg.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size', 'actor_learning_rate',
                   'critic_learning_rate', 'tau', 'goal_focused',
                   'num_time_steps_train', 'avg_score_train',
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
                                for goal in list({None, env_goal}):
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
                                        heuristic_func=heuristic_func, **args
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
                                        'goal_focused': 'Yes' if goal else 'No'
                                    }

                                    for key in result:
                                        new_row.update({key: result[key]})

                                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td3_heuristics(env, env_name, heuristic_func, env_goal, **args):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_heuristics_td3.csv'.format(env_name))

    result_cols = ['learning_type', 'use_model_only', 'batch_size', 'hidden_layer_size', 'actor_learning_rate',
                   'critic_learning_rate', 'tau', 'goal_focused',
                   'num_time_steps_train', 'avg_score_train',
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
                                for goal in list({None, env_goal}):
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
                                        heuristic_func=heuristic_func, **args
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
                                        'goal_focused': 'Yes' if goal else 'No'
                                    }

                                    for key in result:
                                        new_row.update({key: result[key]})

                                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_heuristics(env, env_name, heuristic_func, penalty=0, env_goal=None, **args):
    run_decision_tree_heuristics(env, env_name, heuristic_func, penalty, **args)
    if type(env.action_space) == Discrete:
        run_td_epsilon_greedy_heuristics(env, env_name, heuristic_func, penalty, env_goal, **args)
        run_dueling_td_epsilon_greedy_heuristics(env, env_name, heuristic_func, penalty, env_goal, **args)
        run_td_softmax_heuristics(env, env_name, heuristic_func, penalty, env_goal, **args)
        run_dueling_td_softmax_heuristics(env, env_name, heuristic_func, penalty, env_goal, **args)
        run_td_ucb_heuristics(env, env_name, heuristic_func, penalty, env_goal, **args)
        run_dueling_td_ucb_heuristics(env, env_name, heuristic_func, penalty, env_goal, **args)
        if penalty > 0:
            run_td_thompson_sampling_heuristics(env, env_name, heuristic_func, penalty, env_goal, **args)
            run_dueling_td_thompson_sampling_heuristics(env, env_name, heuristic_func, penalty, env_goal, **args)
    else:
        run_ddpg_heuristics(env, env_name, heuristic_func, env_goal, **args)
        run_td3_heuristics(env, env_name, heuristic_func, env_goal, **args)
