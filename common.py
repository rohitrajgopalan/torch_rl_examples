import os

import pandas as pd

import numpy as np
import gym

from gym.spaces import Box, Discrete

import torch_rl.dueling_td.main
import torch_rl.td.main
import torch_rl.ddpg.main
import torch_rl.td3.main
from torch_rl.utils.types import NetworkOptimizer, TDAlgorithmType, PolicyType


def derive_hidden_layer_size(env, batch_size):
    if type(env.observation_space) == Box:
        return env.observation_space.shape[0] * batch_size
    else:
        return batch_size


def is_observation_space_not_well_defined(env):
    if type(env.observation_space) == Box:
        return (env.observation_space.low == -np.inf).any() or (env.observation_space.high == np.inf).any()
    else:
        return False


class NormalizedStates(gym.ObservationWrapper):
    def observation(self, observation):
        if type(self.observation_space) == Box:
            return (observation - self.observation_space.low) / (
                    self.observation_space.high - self.observation_space.low)
        elif type(self.observation_space) == Discrete:
            return (observation + 1) / self.observation_space.n
        else:
            return observation


class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


def run_td_epsilon_greedy(env, env_name, penalty, env_goal=None, env_move_matrix=None):
    n_games = (500, 50)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_td_epsilon_greedy.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'replay', 'optimizer', 'learning_rate', 'goal_focused',
                   'using_move_matrix',
                   'is_double', 'algorithm_type', 'enable_decay', 'epsilon_start',
                   'num_time_steps_train', 'avg_score_train',
                   'num_actions_blocked_train',
                   'num_time_steps_test', 'avg_score_test', 'num_actions_blocked_test', 'avg_loss']

    is_observation_space_well_defined = not is_observation_space_not_well_defined(env)
    if is_observation_space_well_defined:
        result_cols.insert(0, 'normalize_state')

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}
    normalize_state_flags = [False, True] if is_observation_space_well_defined else [False]
    enable_action_blocker_flags = [False, True] if penalty > 0 else [False]
    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in enable_action_blocker_flags:
                for normalize_state in normalize_state_flags:
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                hidden_layer_sizes = [derive_hidden_layer_size(env, batch_size),
                                                      64, 128, 256, 512]
                                hidden_layer_sizes = list(set(hidden_layer_sizes))
                                for hidden_layer_size in hidden_layer_sizes:
                                    for goal in list({None, env_goal}):
                                        for enable_decay in [False, True]:
                                            epsilons = [1.0] if enable_decay \
                                                else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                                            for epsilon in epsilons:
                                                policy_args.update({'eps_start': epsilon, 'enable_decay': enable_decay})
                                                for using_move_matrix in list({False, env_move_matrix is not None}):
                                                    if using_move_matrix:
                                                        policy_args.update({'move_matrix': env_move_matrix})
                                                    if normalize_state:
                                                        env = NormalizedStates(env)
                                                        if goal:
                                                            goal = env.observation(goal)
                                                    network_optimizer_args = {
                                                        'learning_rate': learning_rate
                                                    }
                                                    result = torch_rl.td.main.run(
                                                        env=env, n_games=n_games, gamma=0.99,
                                                        mem_size=1000,
                                                        batch_size=batch_size,
                                                        fc_dims=hidden_layer_size,
                                                        optimizer_type=optimizer_type,
                                                        replace=1000,
                                                        optimizer_args=network_optimizer_args,
                                                        enable_action_blocking=enable_action_blocker,
                                                        min_penalty=penalty,
                                                        goal=goal,
                                                        is_double=is_double,
                                                        algorithm_type=algorithm_type,
                                                        policy_type=PolicyType.EPSILON_GREEDY,
                                                        policy_args=policy_args)
    
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
                                                        'using_move_matrix': 'Yes' if using_move_matrix else 'No'
                                                    }
                                                    for key in result:
                                                        new_row.update({key: result[key]})
    
                                                    if is_observation_space_well_defined:
                                                        new_row.update(
                                                            {'normalize_state': 'Yes' if normalize_state else 'No'})
                                                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td_softmax(env, env_name, penalty, env_goal=None, env_move_matrix=None):
    n_games = (500, 50)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_td_epsilon_greedy.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'replay', 'optimizer', 'learning_rate', 'goal_focused',
                   'using_move_matrix',
                   'is_double', 'algorithm_type', 'tau', 'num_time_steps_train', 'avg_score_train',
                   'num_actions_blocked_train',
                   'num_time_steps_test', 'avg_score_test', 'num_actions_blocked_test', 'avg_loss']

    is_observation_space_well_defined = not is_observation_space_not_well_defined(env)
    if is_observation_space_well_defined:
        result_cols.insert(0, 'normalize_state')

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}

    normalize_state_flags = [False, True] if is_observation_space_well_defined else [False]
    enable_action_blocker_flags = [False, True] if penalty > 0 else [False]
    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in enable_action_blocker_flags:
                for normalize_state in normalize_state_flags:
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                hidden_layer_sizes = [derive_hidden_layer_size(env, batch_size),
                                                      64, 128, 256, 512]
                                hidden_layer_sizes = list(set(hidden_layer_sizes))
                                for hidden_layer_size in hidden_layer_sizes:
                                    for goal in list({None, env_goal}):
                                        for tau in [0.0001, 0.001, 0.1, 1.0, 10.0]:
                                            policy_args.update({'tau': tau})
                                            for using_move_matrix in list({False, env_move_matrix is not None}):
                                                if using_move_matrix:
                                                    policy_args.update({'move_matrix': env_move_matrix})
                                                if normalize_state:
                                                    env = NormalizedStates(env)
                                                    if goal:
                                                        goal = env.observation(goal)
                                                network_optimizer_args = {
                                                    'learning_rate': learning_rate
                                                }
                                                result = torch_rl.td.main.run(
                                                    env=env, n_games=n_games, gamma=0.99,
                                                    mem_size=1000,
                                                    batch_size=batch_size,
                                                    fc_dims=hidden_layer_size,
                                                    optimizer_type=optimizer_type,
                                                    replace=1000,
                                                    optimizer_args=network_optimizer_args,
                                                    enable_action_blocking=enable_action_blocker,
                                                    min_penalty=penalty,
                                                    goal=goal,
                                                    is_double=is_double,
                                                    algorithm_type=algorithm_type,
                                                    policy_type=PolicyType.SOFTMAX,
                                                    policy_args=policy_args)

                                                new_row = {
                                                    'batch_size': batch_size,
                                                    'hidden_layer_size': hidden_layer_size,
                                                    'algorithm_type': algorithm_type,
                                                    'optimizer': optimizer_type.name.lower(),
                                                    'learning_rate': learning_rate,
                                                    'goal_focused': 'Yes' if goal else 'No',
                                                    'is_double': 'Yes' if is_double else 'No',
                                                    'tau': tau,
                                                    'using_move_matrix': 'Yes' if using_move_matrix else 'No'
                                                }
                                                for key in result:
                                                    new_row.update({key: result[key]})

                                                if is_observation_space_well_defined:
                                                    new_row.update(
                                                        {'normalize_state': 'Yes' if normalize_state else 'No'})
                                                results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td_ucb(env, env_name, penalty, env_goal=None, env_move_matrix=None):
    n_games = (500, 50)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results', '{0}_td_ucb.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'replay', 'optimizer', 'learning_rate', 'goal_focused',
                   'using_move_matrix', 'is_double', 'algorithm_type',
                   'num_time_steps_train', 'avg_score_train', 'num_actions_blocked_train',
                   'num_time_steps_test', 'avg_score_test', 'num_actions_blocked_test', 'avg_loss']

    is_observation_space_well_defined = not is_observation_space_not_well_defined(env)
    if is_observation_space_well_defined:
        result_cols.insert(0, 'normalize_state')

    results = pd.DataFrame(columns=result_cols)

    policy_args = {'confidence_factor': 2}

    normalize_state_flags = [False, True] if is_observation_space_well_defined else [False]
    enable_action_blocker_flags = [False, True] if penalty > 0 else [False]
    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in enable_action_blocker_flags:
                for normalize_state in normalize_state_flags:
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                hidden_layer_sizes = [derive_hidden_layer_size(env, batch_size),
                                                      64, 128, 256, 512]
                                hidden_layer_sizes = list(set(hidden_layer_sizes))
                                for hidden_layer_size in hidden_layer_sizes:
                                    for goal in list({None, env_goal}):
                                        for using_move_matrix in list({False, env_move_matrix is not None}):
                                            if using_move_matrix:
                                                policy_args.update({'move_matrix': env_move_matrix})
                                            if normalize_state:
                                                env = NormalizedStates(env)
                                                if goal:
                                                    goal = env.observation(goal)
                                            network_optimizer_args = {
                                                'learning_rate': learning_rate
                                            }
                                            result = torch_rl.td.main.run(
                                                env=env, n_games=n_games, gamma=0.99,
                                                mem_size=1000,
                                                batch_size=batch_size,
                                                fc_dims=hidden_layer_size,
                                                optimizer_type=optimizer_type,
                                                replace=1000,
                                                optimizer_args=network_optimizer_args,
                                                enable_action_blocking=enable_action_blocker,
                                                min_penalty=penalty,
                                                goal=goal,
                                                is_double=is_double,
                                                algorithm_type=algorithm_type,
                                                policy_type=PolicyType.UCB,
                                                policy_args=policy_args)

                                            new_row = {
                                                'batch_size': batch_size,
                                                'hidden_layer_size': hidden_layer_size,
                                                'algorithm_type': algorithm_type,
                                                'optimizer': optimizer_type.name.lower(),
                                                'learning_rate': learning_rate,
                                                'goal_focused': 'Yes' if goal else 'No',
                                                'is_double': 'Yes' if is_double else 'No',
                                                'using_move_matrix': 'Yes' if using_move_matrix else 'No'
                                            }
                                            for key in result:
                                                new_row.update({key: result[key]})

                                            if is_observation_space_well_defined:
                                                new_row.update({'normalize_state': 'Yes' if normalize_state else 'No'})
                                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td_thompson_sampling(env, env_name, penalty, env_goal=None, env_move_matrix=None):
    assert penalty > 0
    n_games = (500, 50)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_td_thompson_sampling.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'replay', 'optimizer', 'learning_rate', 'goal_focused',
                   'using_move_matrix', 'is_double', 'algorithm_type',
                   'num_time_steps_train', 'avg_score_train', 'num_actions_blocked_train',
                   'num_time_steps_test', 'avg_score_test', 'num_actions_blocked_test', 'avg_loss']

    is_observation_space_well_defined = not is_observation_space_not_well_defined(env)
    if is_observation_space_well_defined:
        result_cols.insert(0, 'normalize_state')

    results = pd.DataFrame(columns=result_cols)

    policy_args = {
        'min_penalty': penalty
    }

    normalize_state_flags = [False, True] if is_observation_space_well_defined else [False]
    enable_action_blocker_flags = [False, True] if penalty > 0 else [False]
    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in enable_action_blocker_flags:
                for normalize_state in normalize_state_flags:
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                hidden_layer_sizes = [derive_hidden_layer_size(env, batch_size),
                                                      64, 128, 256, 512]
                                hidden_layer_sizes = list(set(hidden_layer_sizes))
                                for hidden_layer_size in hidden_layer_sizes:
                                    for goal in list({None, env_goal}):
                                        for using_move_matrix in list({False, env_move_matrix is not None}):
                                            if using_move_matrix:
                                                policy_args.update({'move_matrix': env_move_matrix})
                                            if normalize_state:
                                                env = NormalizedStates(env)
                                                if goal:
                                                    goal = env.observation(goal)
                                            network_optimizer_args = {
                                                'learning_rate': learning_rate
                                            }
                                            result = torch_rl.td.main.run(
                                                env=env, n_games=n_games, gamma=0.99,
                                                mem_size=1000,
                                                batch_size=batch_size,
                                                fc_dims=hidden_layer_size,
                                                optimizer_type=optimizer_type,
                                                replace=1000,
                                                optimizer_args=network_optimizer_args,
                                                enable_action_blocking=enable_action_blocker,
                                                min_penalty=penalty,
                                                goal=goal,
                                                is_double=is_double,
                                                algorithm_type=algorithm_type,
                                                policy_type=PolicyType.THOMPSON_SAMPLING,
                                                policy_args=policy_args)

                                            new_row = {
                                                'batch_size': batch_size,
                                                'hidden_layer_size': hidden_layer_size,
                                                'algorithm_type': algorithm_type,
                                                'optimizer': optimizer_type.name.lower(),
                                                'learning_rate': learning_rate,
                                                'goal_focused': 'Yes' if goal else 'No',
                                                'is_double': 'Yes' if is_double else 'No',
                                                'using_move_matrix': 'Yes' if using_move_matrix else 'No'
                                            }
                                            for key in result:
                                                new_row.update({key: result[key]})

                                            if is_observation_space_well_defined:
                                                new_row.update({'normalize_state': 'Yes' if normalize_state else 'No'})
                                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_epsilon_greedy(env, env_name, penalty, env_goal=None, env_move_matrix=None):
    n_games = (500, 50)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_td_epsilon_greedy.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'replay', 'optimizer', 'learning_rate', 'goal_focused',
                   'using_move_matrix', 'is_double', 'algorithm_type', 'enable_decay', 'epsilon_start',
                   'num_time_steps_train', 'avg_score_train', 'num_actions_blocked_train',
                   'num_time_steps_test', 'avg_score_test', 'num_actions_blocked_test', 'avg_loss']

    is_observation_space_well_defined = not is_observation_space_not_well_defined(env)
    if is_observation_space_well_defined:
        result_cols.insert(0, 'normalize_state')

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}

    normalize_state_flags = [False, True] if is_observation_space_well_defined else [False]
    enable_action_blocker_flags = [False, True] if penalty > 0 else [False]
    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in enable_action_blocker_flags:
                for normalize_state in normalize_state_flags:
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                hidden_layer_sizes = [derive_hidden_layer_size(env, batch_size),
                                                      64, 128, 256, 512]
                                hidden_layer_sizes = list(set(hidden_layer_sizes))
                                for hidden_layer_size in hidden_layer_sizes:
                                    for goal in list({None, env_goal}):
                                        for enable_decay in [False, True]:
                                            epsilons = [1.0] if enable_decay \
                                                else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                                            for epsilon in epsilons:
                                                policy_args.update({'eps_start': epsilon, 'enable_decay': enable_decay})
                                                for using_move_matrix in list({False, env_move_matrix is not None}):
                                                    if using_move_matrix:
                                                        policy_args.update({'move_matrix': env_move_matrix})
                                                    if normalize_state:
                                                        env = NormalizedStates(env)
                                                        if goal:
                                                            goal = env.observation(goal)
                                                    network_optimizer_args = {
                                                        'learning_rate': learning_rate
                                                    }
                                                    result = torch_rl.dueling_td.main.run(
                                                        env=env, n_games=n_games, gamma=0.99,
                                                        mem_size=1000,
                                                        batch_size=batch_size,
                                                        fc_dims=hidden_layer_size,
                                                        optimizer_type=optimizer_type,
                                                        replace=1000,
                                                        optimizer_args=network_optimizer_args,
                                                        enable_action_blocking=enable_action_blocker,
                                                        min_penalty=penalty,
                                                        goal=goal,
                                                        is_double=is_double,
                                                        algorithm_type=algorithm_type,
                                                        policy_type=PolicyType.EPSILON_GREEDY,
                                                        policy_args=policy_args)

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
                                                        'using_move_matrix': 'Yes' if using_move_matrix else 'No'
                                                    }
                                                    for key in result:
                                                        new_row.update({key: result[key]})

                                                    if is_observation_space_well_defined:
                                                        new_row.update(
                                                            {'normalize_state': 'Yes' if normalize_state else 'No'})
                                                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_softmax(env, env_name, penalty, env_goal=None, env_move_matrix=None):
    n_games = (500, 50)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_td_epsilon_greedy.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'replay', 'optimizer', 'learning_rate', 'goal_focused',
                   'using_move_matrix', 'is_double', 'algorithm_type', 'tau', 'num_time_steps_train', 'avg_score_train',
                   'num_actions_blocked_train',
                   'num_time_steps_test', 'avg_score_test', 'num_actions_blocked_test', 'avg_loss']

    is_observation_space_well_defined = not is_observation_space_not_well_defined(env)
    if is_observation_space_well_defined:
        result_cols.insert(0, 'normalize_state')

    results = pd.DataFrame(columns=result_cols)

    policy_args = {}

    normalize_state_flags = [False, True] if is_observation_space_well_defined else [False]
    enable_action_blocker_flags = [False, True] if penalty > 0 else [False]
    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in enable_action_blocker_flags:
                for normalize_state in normalize_state_flags:
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                hidden_layer_sizes = [derive_hidden_layer_size(env, batch_size),
                                                      64, 128, 256, 512]
                                hidden_layer_sizes = list(set(hidden_layer_sizes))
                                for hidden_layer_size in hidden_layer_sizes:
                                    for goal in list({None, env_goal}):
                                        for tau in [0.0001, 0.001, 0.1, 1.0, 10.0]:
                                            policy_args.update({'tau': tau})
                                            for using_move_matrix in list({False, env_move_matrix is not None}):
                                                if using_move_matrix:
                                                    policy_args.update({'move_matrix': env_move_matrix})
                                                if normalize_state:
                                                    env = NormalizedStates(env)
                                                    if goal:
                                                        goal = env.observation(goal)
                                                network_optimizer_args = {
                                                    'learning_rate': learning_rate
                                                }
                                                result = torch_rl.dueling_td.main.run(
                                                    env=env, n_games=n_games, gamma=0.99,
                                                    mem_size=1000,
                                                    batch_size=batch_size,
                                                    fc_dims=hidden_layer_size,
                                                    optimizer_type=optimizer_type,
                                                    replace=1000,
                                                    optimizer_args=network_optimizer_args,
                                                    enable_action_blocking=enable_action_blocker,
                                                    min_penalty=penalty,
                                                    goal=goal,
                                                    is_double=is_double,
                                                    algorithm_type=algorithm_type,
                                                    policy_type=PolicyType.SOFTMAX,
                                                    policy_args=policy_args)

                                                new_row = {
                                                    'batch_size': batch_size,
                                                    'hidden_layer_size': hidden_layer_size,
                                                    'algorithm_type': algorithm_type,
                                                    'optimizer': optimizer_type.name.lower(),
                                                    'learning_rate': learning_rate,
                                                    'goal_focused': 'Yes' if goal else 'No',
                                                    'is_double': 'Yes' if is_double else 'No',
                                                    'tau': tau,
                                                    'using_move_matrix': 'Yes' if using_move_matrix else 'No'
                                                }
                                                for key in result:
                                                    new_row.update({key: result[key]})

                                                if is_observation_space_well_defined:
                                                    new_row.update(
                                                        {'normalize_state': 'Yes' if normalize_state else 'No'})
                                                results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_ucb(env, env_name, penalty, env_goal=None, env_move_matrix=None):
    n_games = (500, 50)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_td_ucb.csv'.format(env_name))
    result_cols = ['batch_size', 'hidden_layer_size', 'replay', 'optimizer', 'learning_rate', 'goal_focused',
                   'using_move_matrix', 'is_double', 'algorithm_type',
                   'num_time_steps_train', 'avg_score_train', 'num_actions_blocked_train',
                   'num_time_steps_test', 'avg_score_test', 'num_actions_blocked_test', 'avg_loss']

    is_observation_space_well_defined = not is_observation_space_not_well_defined(env)

    if is_observation_space_well_defined:
        result_cols.insert(0, 'normalize_state')

    results = pd.DataFrame(columns=result_cols)

    policy_args = {'confidence_factor': 2}

    normalize_state_flags = [False, True] if is_observation_space_well_defined else [False]
    enable_action_blocker_flags = [False, True] if penalty > 0 else [False]
    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in enable_action_blocker_flags:
                for normalize_state in normalize_state_flags:
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                hidden_layer_sizes = [derive_hidden_layer_size(env, batch_size),
                                                      64, 128, 256, 512]
                                hidden_layer_sizes = list(set(hidden_layer_sizes))
                                for hidden_layer_size in hidden_layer_sizes:
                                    for goal in list({None, env_goal}):
                                        for using_move_matrix in list({False, env_move_matrix is not None}):
                                            if using_move_matrix:
                                                policy_args.update({'move_matrix': env_move_matrix})
                                            if normalize_state:
                                                env = NormalizedStates(env)
                                                if goal:
                                                    goal = env.observation(goal)
                                            network_optimizer_args = {
                                                'learning_rate': learning_rate
                                            }
                                            result = torch_rl.dueling_td.main.run(
                                                env=env, n_games=n_games, gamma=0.99,
                                                mem_size=1000,
                                                batch_size=batch_size,
                                                fc_dims=hidden_layer_size,
                                                optimizer_type=optimizer_type,
                                                replace=1000,
                                                optimizer_args=network_optimizer_args,
                                                enable_action_blocking=enable_action_blocker,
                                                min_penalty=penalty,
                                                goal=goal,
                                                is_double=is_double,
                                                algorithm_type=algorithm_type,
                                                policy_type=PolicyType.UCB,
                                                policy_args=policy_args)

                                            new_row = {
                                                'batch_size': batch_size,
                                                'hidden_layer_size': hidden_layer_size,
                                                'algorithm_type': algorithm_type,
                                                'optimizer': optimizer_type.name.lower(),
                                                'learning_rate': learning_rate,
                                                'goal_focused': 'Yes' if goal else 'No',
                                                'is_double': 'Yes' if is_double else 'No',
                                                'using_move_matrix': 'Yes' if using_move_matrix else 'No'
                                            }
                                            for key in result:
                                                new_row.update({key: result[key]})

                                            if is_observation_space_well_defined:
                                                new_row.update({'normalize_state': 'Yes' if normalize_state else 'No'})
                                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_dueling_td_thompson_sampling(env, env_name, penalty, env_goal=None, env_move_matrix=None):
    assert penalty > 0
    n_games = (500, 50)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_td_thompson_sampling.csv'.format(env_name))
    result_cols = ['batch_size', 'hidden_layer_size', 'replay', 'optimizer', 'learning_rate', 'goal_focused',
                   'using_move_matrix', 'is_double', 'algorithm_type',
                   'num_time_steps_train', 'avg_score_train', 'num_actions_blocked_train',
                   'num_time_steps_test', 'avg_score_test', 'num_actions_blocked_test', 'avg_loss']

    is_observation_space_well_defined = not is_observation_space_not_well_defined(env)

    if is_observation_space_well_defined:
        result_cols.insert(0, 'normalize_state')

    results = pd.DataFrame(columns=result_cols)

    policy_args = {
        'min_penalty': penalty
    }

    normalize_state_flags = [False, True] if is_observation_space_well_defined else [False]
    enable_action_blocker_flags = [False, True] if penalty > 0 else [False]
    for is_double in [False, True]:
        for algorithm_type in TDAlgorithmType.all():
            for enable_action_blocker in enable_action_blocker_flags:
                for normalize_state in normalize_state_flags:
                    for batch_size in [32, 64, 128]:
                        for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                            for learning_rate in [0.001, 0.0001]:
                                hidden_layer_sizes = [derive_hidden_layer_size(env, batch_size),
                                                      64, 128, 256, 512]
                                hidden_layer_sizes = list(set(hidden_layer_sizes))
                                for hidden_layer_size in hidden_layer_sizes:
                                    for goal in list({None, env_goal}):
                                        for using_move_matrix in list({False, env_move_matrix is not None}):
                                            if using_move_matrix:
                                                policy_args.update({'move_matrix': env_move_matrix})
                                            if normalize_state:
                                                env = NormalizedStates(env)
                                                if goal:
                                                    goal = env.observation(goal)
                                            network_optimizer_args = {
                                                'learning_rate': learning_rate
                                            }
                                            result = torch_rl.dueling_td.main.run(
                                                env=env, n_games=n_games, gamma=0.99,
                                                mem_size=1000,
                                                batch_size=batch_size,
                                                fc_dims=hidden_layer_size,
                                                optimizer_type=optimizer_type,
                                                replace=1000,
                                                optimizer_args=network_optimizer_args,
                                                enable_action_blocking=enable_action_blocker,
                                                min_penalty=penalty,
                                                goal=goal,
                                                is_double=is_double,
                                                algorithm_type=algorithm_type,
                                                policy_type=PolicyType.THOMPSON_SAMPLING,
                                                policy_args=policy_args)

                                            new_row = {
                                                'batch_size': batch_size,
                                                'hidden_layer_size': hidden_layer_size,
                                                'algorithm_type': algorithm_type,
                                                'optimizer': optimizer_type.name.lower(),
                                                'learning_rate': learning_rate,
                                                'goal_focused': 'Yes' if goal else 'No',
                                                'is_double': 'Yes' if is_double else 'No',
                                                'using_move_matrix': 'Yes' if using_move_matrix else 'No'
                                            }
                                            for key in result:
                                                new_row.update({key: result[key]})

                                            if is_observation_space_well_defined:
                                                new_row.update({'normalize_state': 'Yes' if normalize_state else 'No'})
                                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_all_td_methods(env, env_name, penalty, env_goal=None, env_move_matrix=None):
    print('Running', env_name)
    run_td_epsilon_greedy(env, env_name, penalty, env_goal, env_move_matrix)
    run_dueling_td_epsilon_greedy(env, env_name, penalty, env_goal, env_move_matrix)
    run_td_softmax(env, env_name, penalty, env_goal, env_move_matrix)
    run_dueling_td_softmax(env, env_name, penalty, env_goal, env_move_matrix)
    run_td_ucb(env, env_name, penalty, env_goal, env_move_matrix)
    run_dueling_td_ucb(env, env_name, penalty, env_goal, env_move_matrix)
    if penalty > 0:
        run_td_thompson_sampling(env, env_name, penalty, env_goal, env_move_matrix)
        run_dueling_td_thompson_sampling(env, env_name, penalty, env_goal, env_move_matrix)


def run_ddpg(env, env_name):
    n_games = (100, 10)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_ddpg.csv'.format(env_name))

    result_cols = ['normalize_actions', 'batch_size', 'hidden_layer_size', 'replay', 'actor_learning_rate',
                   'critic_learning_rate', 'tau',
                   'num_time_steps_train', 'avg_score_train', 'num_time_steps_test', 'avg_score_test',
                   'avg_policy_loss', 'avg_value_loss']

    is_observation_space_well_defined = not is_observation_space_not_well_defined(env)

    if is_observation_space_well_defined:
        result_cols.insert(0, 'normalize_states')

    results = pd.DataFrame(
        columns=result_cols)

    normalize_state_flags = [False, True] if is_observation_space_well_defined else [False]
    for normalize_state in normalize_state_flags:
        for normalize_actions in [False, True]:
            for batch_size in [64, 128]:
                hidden_layer_sizes = [64, 128, 256, 300, 400, 512,
                                      derive_hidden_layer_size(env, batch_size)]
                hidden_layer_sizes = list(set(hidden_layer_sizes))
                for hidden_layer_size in hidden_layer_sizes:
                    for randomized in [False, True]:
                        for actor_learning_rate in [0.001, 0.0001]:
                            for critic_learning_rate in [0.001, 0.0001]:
                                for tau in [1e-2, 1e-3]:
                                    if normalize_actions:
                                        env = NormalizedActions(env)
                                    if normalize_state:
                                        env = NormalizedStates(env)
                                    actor_optimizer_args = {
                                        'learning_rate': actor_learning_rate
                                    }
                                    critic_optimizer_args = {
                                        'learning_rate': critic_learning_rate
                                    }
                                    print('Running instance of DDPG: Actions {0}normalized, {1} replay, batch size of '
                                          '{2}, '
                                          'hidden layer size of {3}, '
                                          'actor learning rate of {4}, critic learning rate of {5} and tau {6} with '
                                          '{0}normalized states'
                                          .format('' if normalize_actions else 'un',
                                                  'Randomized' if randomized else 'Sequenced', batch_size,
                                                  hidden_layer_size,
                                                  actor_learning_rate, critic_learning_rate, tau,
                                                  '' if normalize_state else 'un'))
                                    num_time_steps_train, avg_score_train, num_time_steps_test, avg_score_test, avg_policy_loss, avg_critic_loss = torch_rl.ddpg.main.run(
                                        env=env, n_games=n_games, tau=tau, fc_dims=hidden_layer_size,
                                        batch_size=batch_size, randomized=randomized,
                                        actor_optimizer_type=NetworkOptimizer.ADAM,
                                        critic_optimizer_type=NetworkOptimizer.ADAM,
                                        actor_optimizer_args=actor_optimizer_args,
                                        critic_optimizer_args=critic_optimizer_args
                                    )

                                    new_row = {
                                        'normalize_actions': 'Yes' if normalize_actions else 'No',
                                        'batch_size': batch_size,
                                        'hidden_layer_size': hidden_layer_size,
                                        'replay': 'Randomized' if randomized else 'Sequenced',
                                        'actor_learning_rate': actor_learning_rate,
                                        'critic_learning_rate': critic_learning_rate,
                                        'tau': tau,
                                        'num_time_steps_train': num_time_steps_train,
                                        'avg_score_train': round(avg_score_train, 5),
                                        'num_time_steps_test': num_time_steps_test,
                                        'avg_score_test': round(avg_score_test, 5),
                                        'avg_policy_loss': round(avg_policy_loss, 5),
                                        'avg_value_loss': round(avg_critic_loss, 5)
                                    }

                                    if is_observation_space_well_defined:
                                        new_row.update({'normalize_state': 'Yes' if normalize_state else 'No'})

                                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_td3(env, env_name):
    n_games = 75

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_td3.csv'.format(env_name))

    result_cols = ['normalize_actions', 'batch_size', 'hidden_layer_size', 'replay', 'actor_learning_rate',
                   'critic_learning_rate', 'tau',
                   'num_time_steps_train', 'avg_score_train', 'num_time_steps_test', 'avg_score_test',
                   'avg_policy_loss', 'avg_value1_loss',
                   'avg_value2_loss']

    is_observation_space_well_defined = not is_observation_space_not_well_defined(env)

    if is_observation_space_well_defined:
        result_cols.insert(0, 'normalize_states')

    results = pd.DataFrame(
        columns=result_cols)

    actor_optimizer_args = {
        'learning_rate': 1e-3
    }
    critic_optimizer_args = {
        'learning_rate': 1e-3
    }

    normalize_state_flags = [False, True] if is_observation_space_well_defined else [False]
    for normalize_state in normalize_state_flags:
        for normalize_actions in [False, True]:
            for batch_size in [64, 100, 128]:
                hidden_layer_sizes = [64, 128, 256, 300, 400, 512,
                                      derive_hidden_layer_size(env, batch_size)]
                hidden_layer_sizes = list(set(hidden_layer_sizes))
                for hidden_layer_size in hidden_layer_sizes:
                    for randomized in [False, True]:
                        for tau in [0.005, 0.01]:
                            if normalize_actions:
                                env = NormalizedActions(env)
                            if normalize_state:
                                env = NormalizedStates(env)
                            print('Running instance of TD3: Actions {0}normalized, {1} replay, batch size of {2}, '
                                  'hidden layer size of {3}, '
                                  'and tau {4} with {5}normalized states'
                                  .format('' if normalize_actions else 'un',
                                          'Randomized' if randomized else 'Sequenced',
                                          batch_size, hidden_layer_size,
                                          tau, '' if normalize_state else 'un'))
                            num_time_steps_train, avg_score_train, num_time_steps_test, avg_score_test, avg_policy_loss, avg_value1_loss, avg_value2_loss = torch_rl.td3.main.run(
                                env=env, n_games=n_games, tau=tau, fc_dims=hidden_layer_size,
                                batch_size=batch_size, randomized=randomized,
                                actor_optimizer_type=NetworkOptimizer.ADAM,
                                critic_optimizer_type=NetworkOptimizer.ADAM, actor_optimizer_args=actor_optimizer_args,
                                critic_optimizer_args=critic_optimizer_args
                            )

                            new_row = {
                                'normalize_actions': normalize_actions,
                                'batch_size': batch_size,
                                'hidden_layer_size': hidden_layer_size,
                                'replay': 'Randomized' if randomized else 'Sequenced',
                                'tau': tau,
                                'num_time_steps_train': num_time_steps_train,
                                'avg_score_train': round(avg_score_test, 5),
                                'num_time_steps_test': num_time_steps_train,
                                'avg_score_test': round(avg_score_test, 5),
                                'avg_policy_loss': round(avg_policy_loss, 5),
                                'avg_value1_loss': round(avg_value1_loss, 5),
                                'avg_value2_loss': round(avg_value2_loss, 5)
                            }

                            if is_observation_space_well_defined:
                                new_row.update({'normalize_state': 'Yes' if normalize_state else 'No'})

                            results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False, float_format='%.3f')


def run_actor_critic_continuous_methods(env, env_name):
    print('Running', env_name)
    run_ddpg(env, env_name)
    run_td3(env, env_name)
