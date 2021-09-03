import os

import pandas as pd
from torch_rl.td.agent import TDAgent
from torch_rl.dueling_td.agent import DuelingTDAgent
from torch_rl.utils.types import NetworkOptimizer, TDAlgorithmType, PolicyType, LearningType

from petting_zoo.random_legal import RandomLegal

try:
    from run import run_pettingzoo_env
    from utils import develop_memory_for_dt_action_blocker_for_pettingzoo_env

except ImportError:
    from .run import run_pettingzoo_env
    from .utils import develop_memory_for_dt_action_blocker_for_pettingzoo_env


def run_td_epsilon_greedy(env, env_name, penalty):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_td_epsilon_greedy.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate',
                   'assign_priority', 'enable_action_blocker', 'use_ml_for_action_blocker',
                   'is_double', 'algorithm_type', 'enable_decay', 'epsilon_start',
                   'num_time_steps_train', 'avg_score_train',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    action_blocker_memory = develop_memory_for_dt_action_blocker_for_pettingzoo_env(env)

    policy_args = {}

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in list({False, penalty > 0}):
                for batch_size in [32, 64, 128]:
                    for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                        for learning_rate in [0.001, 0.0001]:
                            for hidden_layer_size in [256, 512, 1024]:
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
                                                    is_double=is_double,
                                                    algorithm_type=algorithm_type,
                                                    policy_type=PolicyType.EPSILON_GREEDY,
                                                    policy_args=policy_args,
                                                    assign_priority=assign_priority,
                                                    use_ml_for_action_blocker=use_ml_for_action_blocker,
                                                    action_blocker_memory=action_blocker_memory)

                                                agents = {env.possible_agents[0]: agent,
                                                          env.possible_agents[1]: RandomLegal()}

                                                result = run_pettingzoo_env(env, agents,
                                                                            n_games_train=500, n_games_test=50)

                                                new_row = {
                                                    'batch_size': batch_size,
                                                    'hidden_layer_size': hidden_layer_size,
                                                    'algorithm_type': algorithm_type,
                                                    'optimizer': optimizer_type.name.lower(),
                                                    'learning_rate': learning_rate,
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


def run_td_softmax(env, env_name, penalty):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_td_softmax.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate',
                   'assign_priority', 'enable_action_blocker', 'use_ml_for_action_blocker',
                   'is_double', 'algorithm_type', 'tau', 'num_time_steps_train', 'avg_score_train',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    action_blocker_memory = develop_memory_for_dt_action_blocker_for_pettingzoo_env(env)
    policy_args = {}

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in list({False, penalty > 0}):
                for batch_size in [32, 64, 128]:
                    for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                        for learning_rate in [0.001, 0.0001]:
                            for hidden_layer_size in [256, 512, 1024]:
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
                                                is_double=is_double,
                                                algorithm_type=algorithm_type,
                                                policy_type=PolicyType.SOFTMAX,
                                                policy_args=policy_args,
                                                assign_priority=assign_priority,
                                                use_ml_for_action_blocker=use_ml_for_action_blocker,
                                                action_blocker_memory=action_blocker_memory)

                                            agents = {env.possible_agents[0]: agent,
                                                      env.possible_agents[1]: RandomLegal()}

                                            result = run_pettingzoo_env(env, agents,
                                                                        n_games_train=500, n_games_test=50)

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
                                                'use_ml_for_action_blocker': 'Yes' if use_ml_for_action_blocker else 'No'
                                            }
                                            for key in result:
                                                new_row.update({key: result[key]})

                                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td_ucb(env, env_name, penalty):
    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results', '{0}_td_ucb.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate',
                   'assign_priority', 'is_double', 'algorithm_type',
                   'enable_action_blocker', 'use_ml_for_action_blocker',
                   'num_time_steps_train', 'avg_score_train',
                   'num_time_steps_test', 'avg_score_test']

    results = pd.DataFrame(columns=result_cols)

    action_blocker_memory = develop_memory_for_dt_action_blocker_for_pettingzoo_env(env)

    policy_args = {'confidence_factor': 2}

    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in list({False, penalty > 0}):
                for batch_size in [32, 64, 128]:
                    for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                        for learning_rate in [0.001, 0.0001]:
                            for hidden_layer_size in [256, 512, 1024]:
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
                                            is_double=is_double,
                                            algorithm_type=algorithm_type,
                                            policy_type=PolicyType.UCB,
                                            policy_args=policy_args,
                                            assign_priority=assign_priority,
                                            use_ml_for_action_blocker=use_ml_for_action_blocker,
                                            action_blocker_memory=action_blocker_memory)

                                        agents = {env.possible_agents[0]: agent,
                                                  env.possible_agents[1]: RandomLegal()}

                                        result = run_pettingzoo_env(env, agents,
                                                                    n_games_train=500, n_games_test=50)

                                        new_row = {
                                            'batch_size': batch_size,
                                            'hidden_layer_size': hidden_layer_size,
                                            'algorithm_type': algorithm_type,
                                            'optimizer': optimizer_type.name.lower(),
                                            'learning_rate': learning_rate,
                                            'is_double': 'Yes' if is_double else 'No',
                                            'assign_priority': 'Yes' if assign_priority else 'No',
                                            'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                            'use_ml_for_action_blocker': 'Yes' if use_ml_for_action_blocker else 'No'
                                        }
                                        for key in result:
                                            new_row.update({key: result[key]})

                                        results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td_thompson_sampling(env, env_name, penalty):
    assert penalty > 0
    action_blocker_memory = develop_memory_for_dt_action_blocker_for_pettingzoo_env(env)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_td_thompson_sampling.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate',
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
                            for hidden_layer_size in [256, 512, 1024]:
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
                                            is_double=is_double,
                                            algorithm_type=algorithm_type,
                                            policy_type=PolicyType.THOMPSON_SAMPLING,
                                            policy_args=policy_args,
                                            assign_priority=assign_priority,
                                            use_ml_for_action_blocker=use_ml_for_action_blocker,
                                            action_blocker_memory=action_blocker_memory)

                                        agents = {env.possible_agents[0]: agent,
                                                  env.possible_agents[1]: RandomLegal()}

                                        result = run_pettingzoo_env(env, agents,
                                                                    n_games_train=500, n_games_test=50)

                                        new_row = {
                                            'batch_size': batch_size,
                                            'hidden_layer_size': hidden_layer_size,
                                            'algorithm_type': algorithm_type,
                                            'optimizer': optimizer_type.name.lower(),
                                            'learning_rate': learning_rate,
                                            'is_double': 'Yes' if is_double else 'No',
                                            'assign_priority': 'Yes' if assign_priority else 'No',
                                            'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                            'use_ml_for_action_blocker': 'Yes' if use_ml_for_action_blocker else 'No'
                                        }
                                        for key in result:
                                            new_row.update({key: result[key]})

                                        results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_epsilon_greedy(env, env_name, penalty):
    action_blocker_memory = develop_memory_for_dt_action_blocker_for_pettingzoo_env(env)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_td_epsilon_greedy.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate',
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
                            for hidden_layer_size in [256, 512, 1024]:
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
                                                    is_double=is_double,
                                                    algorithm_type=algorithm_type,
                                                    policy_type=PolicyType.EPSILON_GREEDY,
                                                    policy_args=policy_args,
                                                    assign_priority=assign_priority,
                                                    use_ml_for_action_blocker=use_ml_for_action_blocker,
                                                    action_blocker_memory=action_blocker_memory)

                                                agents = {env.possible_agents[0]: agent,
                                                          env.possible_agents[1]: RandomLegal()}

                                                result = run_pettingzoo_env(env, agents,
                                                                            n_games_train=500, n_games_test=50)

                                                new_row = {
                                                    'batch_size': batch_size,
                                                    'hidden_layer_size': hidden_layer_size,
                                                    'algorithm_type': algorithm_type,
                                                    'optimizer': optimizer_type.name.lower(),
                                                    'learning_rate': learning_rate,
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


def run_dueling_td_softmax(env, env_name, penalty):
    action_blocker_memory = develop_memory_for_dt_action_blocker_for_pettingzoo_env(env)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_td_softmax.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate',
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
                            for hidden_layer_size in [256, 512, 1024]:
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
                                                is_double=is_double,
                                                algorithm_type=algorithm_type,
                                                policy_type=PolicyType.SOFTMAX,
                                                policy_args=policy_args,
                                                assign_priority=assign_priority,
                                                use_ml_for_action_blocker=use_ml_for_action_blocker,
                                                action_blocker_memory=action_blocker_memory)

                                            agents = {env.possible_agents[0]: agent,
                                                      env.possible_agents[1]: RandomLegal()}

                                            result = run_pettingzoo_env(env, agents,
                                                                        n_games_train=500, n_games_test=50)

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
                                                'use_ml_for_action_blocker': 'Yes' if use_ml_for_action_blocker else 'No'
                                            }
                                            for key in result:
                                                new_row.update({key: result[key]})

                                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_ucb(env, env_name, penalty):
    action_blocker_memory = develop_memory_for_dt_action_blocker_for_pettingzoo_env(env)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_td_ucb.csv'.format(env_name))
    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate',
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
                            for hidden_layer_size in [256, 512, 1024]:
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
                                            'is_double': 'Yes' if is_double else 'No',
                                            'assign_priority': 'Yes' if assign_priority else 'No',
                                            'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                            'use_ml_for_action_blocker': 'Yes' if use_ml_for_action_blocker else 'No'
                                        }

                                        agents = {env.possible_agents[0]: agent,
                                                  env.possible_agents[1]: RandomLegal()}

                                        result = run_pettingzoo_env(env, agents,
                                                                    n_games_train=500, n_games_test=50)

                                        for key in result:
                                            new_row.update({key: result[key]})

                                        results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_thompson_sampling(env, env_name, penalty):
    assert penalty > 0
    action_blocker_memory = develop_memory_for_dt_action_blocker_for_pettingzoo_env(env)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_td_thompson_sampling.csv'.format(env_name))
    result_cols = ['batch_size', 'hidden_layer_size', 'optimizer', 'learning_rate',
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
                            for hidden_layer_size in [256, 512, 1024]:
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
                                            is_double=is_double,
                                            algorithm_type=algorithm_type,
                                            policy_type=PolicyType.THOMPSON_SAMPLING,
                                            policy_args=policy_args,
                                            assign_priority=assign_priority,
                                            use_ml_for_action_blocker=use_ml_for_action_blocker,
                                            action_blocker_memory=action_blocker_memory)

                                        agents = {env.possible_agents[0]: agent,
                                                  env.possible_agents[1]: RandomLegal()}

                                        result = run_pettingzoo_env(env, agents,
                                                                    n_games_train=500, n_games_test=50)

                                        new_row = {
                                            'batch_size': batch_size,
                                            'hidden_layer_size': hidden_layer_size,
                                            'algorithm_type': algorithm_type,
                                            'optimizer': optimizer_type.name.lower(),
                                            'learning_rate': learning_rate,
                                            'is_double': 'Yes' if is_double else 'No',
                                            'assign_priority': 'Yes' if assign_priority else 'No',
                                            'enable_action_blocker': 'Yes' if enable_action_blocker else 'No',
                                            'use_ml_for_action_blocker': 'Yes' if use_ml_for_action_blocker else 'No'
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
