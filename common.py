import os

import pandas as pd

import numpy as np
import gym

from gym.spaces import Box, Discrete

import torch_rl.dqn.main
import torch_rl.ddqn.main
import torch_rl.dueling_dqn.main
import torch_rl.dueling_ddqn.main
import torch_rl.reinforce.main
import torch_rl.actor_critic.main
import torch_rl.ddpg.main
import torch_rl.td3.main
import torch_rl.sac.main
import torch_rl.ppo.main
from torch_rl.utils.types import NetworkOptimizer


def derive_hidden_layer_size(env, batch_size):
    if getattr(env.observation_space, 'shape'):
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
            return (observation - self.observation_space.low) / (self.observation_space.high - self.observation_space.low)
        elif type(self.observation_space) == Discrete:
            return (observation + 1)/self.observation_space.n
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


def run_dqn(env, env_name, penalty, env_goal=None):
    n_games = (500, 50)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results', '{0}_dqn.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'replay', 'optimizer', 'learning_rate', 'goal_focused',
                   'num_time_steps_train', 'avg_score_train', 'action_blocker_precision_train',
                   'action_blocker_recall_train', 'num_actions_blocked_train',
                   'num_time_steps_test', 'avg_score_test',
                   'action_blocker_precision_test', 'action_blocker_recall_test',
                   'num_actions_blocked_test',
                   'avg_loss']

    is_observation_space_well_defined = not is_observation_space_not_well_defined(env)

    if is_observation_space_well_defined:
        result_cols.insert(0, 'normalize_state')

    results = pd.DataFrame(columns=result_cols)

    normalize_state_flags = [False, True] if is_observation_space_well_defined else [False]
    enable_action_blocker_flags = [False, True] if penalty > 0 else [False]
    for enable_action_blocker in enable_action_blocker_flags:
        for normalize_state in normalize_state_flags:
            for batch_size in [32, 64, 128]:
                for randomized in [False, True]:
                    for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                        for learning_rate in [0.001, 0.0001]:
                            hidden_layer_sizes = [derive_hidden_layer_size(env.observation_space.shape, batch_size),
                                                  64, 128, 256, 512]
                            hidden_layer_sizes = list(set(hidden_layer_sizes))
                            for hidden_layer_size in hidden_layer_sizes:
                                for goal in list({None, env_goal}):
                                    if normalize_state:
                                        env = NormalizedStates(env)
                                        if goal:
                                            goal = env.observation(goal)
                                    network_optimizer_args = {
                                        'learning_rate': learning_rate
                                    }
                                    result = torch_rl.dqn.main.run(
                                        env=env, n_games=n_games, gamma=0.99,
                                        epsilon=1.0, mem_size=1000,
                                        batch_size=batch_size,
                                        fc_dims=hidden_layer_size,
                                        optimizer_type=optimizer_type,
                                        eps_min=0.01,
                                        eps_dec=5e-7,
                                        replace=1000,
                                        optimizer_args=network_optimizer_args,
                                        randomized=randomized,
                                        enable_action_blocking=enable_action_blocker,
                                        min_penalty=penalty,
                                        goal=goal)

                                    new_row = {
                                        'batch_size': batch_size,
                                        'hidden_layer_size': hidden_layer_size,
                                        'replay': 'Randomized' if randomized else 'Sequenced',
                                        'optimizer': optimizer_type.name.lower(),
                                        'learning_rate': learning_rate,
                                        'goal_focused': 'Yes' if goal else 'No'
                                    }
                                    for key in result:
                                        new_row.update({key: result[key]})

                                    if is_observation_space_well_defined:
                                        new_row.update({'normalize_state': 'Yes' if normalize_state else 'No'})
                                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False)


def run_ddqn(env, env_name, penalty, env_goal=None):
    n_games = (500, 50)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results', '{0}_ddqn.csv'.format(env_name))
    result_cols = ['batch_size', 'hidden_layer_size', 'replay', 'optimizer', 'learning_rate', 'goal_focused',
                   'num_time_steps_train', 'avg_score_train', 'action_blocker_precision_train',
                   'action_blocker_recall_train', 'num_actions_blocked_train',
                   'num_time_steps_test', 'avg_score_test',
                   'action_blocker_precision_test', 'action_blocker_recall_test',
                   'num_actions_blocked_test',
                   'avg_loss']

    is_observation_space_well_defined = not is_observation_space_not_well_defined(env)

    if is_observation_space_well_defined:
        result_cols.insert(0, 'normalize_state')

    results = pd.DataFrame(columns=result_cols)

    normalize_state_flags = [False, True] if is_observation_space_well_defined else [False]

    enable_action_blocker_flags = [False, True] if penalty > 0 else [False]
    for enable_action_blocker in enable_action_blocker_flags:
        for normalize_state in normalize_state_flags:
            for batch_size in [32, 64, 128]:
                for randomized in [False, True]:
                    for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                        for learning_rate in [0.001, 0.0001]:
                            hidden_layer_sizes = [derive_hidden_layer_size(env.observation_space.shape, batch_size),
                                                  64, 128, 256, 512]
                            hidden_layer_sizes = list(set(hidden_layer_sizes))
                            for hidden_layer_size in hidden_layer_sizes:
                                for goal in list({None, env_goal}):
                                    if normalize_state:
                                        env = NormalizedStates(env)
                                        if goal:
                                            goal = env.observation(goal)
                                    network_optimizer_args = {
                                        'learning_rate': learning_rate
                                    }
                                    result = torch_rl.dqn.main.run(
                                        env=env, n_games=n_games,
                                        gamma=0.99,
                                        epsilon=1.0,
                                        mem_size=1000,
                                        batch_size=batch_size,
                                        fc_dims=hidden_layer_size,
                                        optimizer_type=optimizer_type,
                                        eps_min=0.01,
                                        eps_dec=5e-7,
                                        replace=1000,
                                        optimizer_args=network_optimizer_args,
                                        randomized=randomized,
                                        enable_action_blocking=enable_action_blocker,
                                        min_penalty=penalty,
                                        goal=goal)

                                    new_row = {
                                        'batch_size': batch_size,
                                        'hidden_layer_size': hidden_layer_size,
                                        'replay': 'Randomized' if randomized else 'Sequenced',
                                        'optimizer': optimizer_type.name.lower(),
                                        'learning_rate': learning_rate,
                                        'goal_focused': 'Yes' if goal else 'No'
                                    }
                                    for key in result:
                                        new_row.update({key: result[key]})

                                    if is_observation_space_well_defined:
                                        new_row.update({'normalize_state': 'Yes' if normalize_state else 'No'})
                                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False)


def run_dueling_dqn(env, env_name, penalty, env_goal=None):
    n_games = (500, 50)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_dqn.csv'.format(env_name))
    result_cols = ['batch_size', 'hidden_layer_size', 'replay', 'optimizer', 'learning_rate', 'goal_focused',
                   'num_time_steps_train', 'avg_score_train', 'action_blocker_precision_train',
                   'action_blocker_recall_train', 'num_actions_blocked_train',
                   'num_time_steps_test', 'avg_score_test',
                   'action_blocker_precision_test', 'action_blocker_recall_test',
                   'num_actions_blocked_test',
                   'avg_loss']

    is_observation_space_well_defined = not is_observation_space_not_well_defined(env)

    if is_observation_space_well_defined:
        result_cols.insert(0, 'normalize_state')

    results = pd.DataFrame(columns=result_cols)

    normalize_state_flags = [False, True] if is_observation_space_well_defined else [False]
    enable_action_blocker_flags = [False, True] if penalty > 0 else [False]
    for enable_action_blocker in enable_action_blocker_flags:
        for normalize_state in normalize_state_flags:
            for batch_size in [32, 64, 128]:
                for randomized in [False, True]:
                    for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                        for learning_rate in [0.001, 0.0001]:
                            hidden_layer_sizes = [derive_hidden_layer_size(env.observation_space.shape, batch_size),
                                                  64, 128, 256, 512]
                            hidden_layer_sizes = list(set(hidden_layer_sizes))
                            for hidden_layer_size in hidden_layer_sizes:
                                for goal in list({None, env_goal}):
                                    if normalize_state:
                                        env = NormalizedStates(env)
                                        if goal:
                                            goal = env.observation(goal)
                                    network_optimizer_args = {
                                        'learning_rate': learning_rate
                                    }
                                    result = torch_rl.dueling_dqn.main.run(
                                        env=env, n_games=n_games,
                                        gamma=0.99,
                                        epsilon=1.0, mem_size=1000,
                                        batch_size=batch_size,
                                        fc_dims=hidden_layer_size,
                                        optimizer_type=optimizer_type,
                                        eps_min=0.01,
                                        eps_dec=5e-7,
                                        replace=1000,
                                        optimizer_args=network_optimizer_args,
                                        randomized=randomized,
                                        enable_action_blocking=enable_action_blocker,
                                        min_penalty=penalty,
                                        goal=goal)

                                    new_row = {
                                        'batch_size': batch_size,
                                        'hidden_layer_size': hidden_layer_size,
                                        'replay': 'Randomized' if randomized else 'Sequenced',
                                        'optimizer': optimizer_type.name.lower(),
                                        'learning_rate': learning_rate,
                                        'goal_focused': 'Yes' if goal else 'No'
                                    }
                                    for key in result:
                                        new_row.update({key: result[key]})
                                    if is_observation_space_well_defined:
                                        new_row.update({'normalize_state': 'Yes' if normalize_state else 'No'})
                                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False)


def run_dueling_ddqn(env, env_name, penalty, env_goal=None):
    n_games = (500, 50)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_ddqn.csv'.format(env_name))

    result_cols = ['batch_size', 'hidden_layer_size', 'replay', 'optimizer', 'learning_rate', 'goal_focused',
                   'num_time_steps_train', 'avg_score_train', 'action_blocker_precision_train',
                   'action_blocker_recall_train', 'num_actions_blocked_train',
                   'num_time_steps_test', 'avg_score_test',
                   'action_blocker_precision_test', 'action_blocker_recall_test',
                   'num_actions_blocked_test',
                   'avg_loss']

    is_observation_space_well_defined = not is_observation_space_not_well_defined(env)

    if is_observation_space_well_defined:
        result_cols.insert(0, 'normalize_state')

    results = pd.DataFrame(columns=result_cols)

    normalize_state_flags = [False, True] if is_observation_space_well_defined else [False]
    enable_action_blocker_flags = [False, True] if penalty > 0 else [False]
    for enable_action_blocker in enable_action_blocker_flags:
        for normalize_state in normalize_state_flags:
            for batch_size in [32, 64, 128]:
                for randomized in [False, True]:
                    for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                        for learning_rate in [0.001, 0.0001]:
                            hidden_layer_sizes = [derive_hidden_layer_size(env.observation_space.shape, batch_size),
                                                  64, 128, 256, 512]
                            hidden_layer_sizes = list(set(hidden_layer_sizes))
                            for hidden_layer_size in hidden_layer_sizes:
                                for goal in list({None, env_goal}):
                                    if normalize_state:
                                        env = NormalizedStates(env)
                                        if goal:
                                            goal = env.observation(goal)
                                    network_optimizer_args = {
                                        'learning_rate': learning_rate
                                    }
                                    result = torch_rl.dueling_ddqn.main.run(
                                        env=env, n_games=n_games,
                                        gamma=0.99,
                                        epsilon=1.0, mem_size=1000,
                                        batch_size=batch_size,
                                        fc_dims=hidden_layer_size,
                                        optimizer_type=optimizer_type,
                                        eps_min=0.01,
                                        eps_dec=5e-7,
                                        replace=1000,
                                        optimizer_args=network_optimizer_args,
                                        randomized=randomized,
                                        enable_action_blocking=enable_action_blocker,
                                        min_penalty=penalty,
                                        goal=goal)

                                    new_row = {
                                        'batch_size': batch_size,
                                        'hidden_layer_size': hidden_layer_size,
                                        'replay': 'Randomized' if randomized else 'Sequenced',
                                        'optimizer': optimizer_type.name.lower(),
                                        'learning_rate': learning_rate,
                                        'goal_focused': 'Yes' if goal else 'No'
                                    }
                                    for key in result:
                                        new_row.update({key: result[key]})

                                    if is_observation_space_well_defined:
                                        new_row.update({'normalize_state': 'Yes' if normalize_state else 'No'})
                                    results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False)


def run_dqn_methods(env, env_name, penalty, env_goal=None):
    print('Running DQN')
    run_dqn(env, env_name, penalty, env_goal)
    print('Running DDQN')
    run_ddqn(env, env_name, penalty, env_goal)
    print('Running Dueling DQN')
    run_dueling_dqn(env, env_name, penalty, env_goal)
    print('Running Dueling DDQN')
    run_dueling_ddqn(env, env_name, penalty, env_goal)


def run_reinforce(env, env_name, penalty, env_goal=None):
    n_games = (500, 50)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_reinforce.csv'.format(env_name))

    result_cols = ['hidden_layer_size', 'learning_rate', 'goal_focused',
                   'num_time_steps_train', 'avg_score_train',
                   'action_blocker_precision_train',
                   'action_blocker_recall_train', 'num_actions_blocked_train',
                   'num_time_steps_test', 'avg_score_test',
                   'action_blocker_precision_test', 'action_blocker_recall_test',
                   'num_actions_blocked_test',
                   'avg_loss']

    is_observation_space_well_defined = not is_observation_space_not_well_defined(env)

    if is_observation_space_well_defined:
        result_cols.insert(0, 'normalize_states')

    results = pd.DataFrame(columns=result_cols)

    normalize_state_flags = [False, True] if is_observation_space_well_defined else [False]
    enable_action_blocker_flags = [False, True] if penalty > 0 else [False]
    for enable_action_blocker in enable_action_blocker_flags:
        for normalize_state in normalize_state_flags:
            for hidden_layer_size in [64, 128, 256, 512]:
                for learning_rate in [0.001, 0.0005]:
                    for goal in list({None, env_goal}):
                        if normalize_state:
                            env = NormalizedStates(env)
                            if goal:
                                goal = env.observation(goal)
                        network_optimizer_args = {
                            'learning_rate': learning_rate
                        }
                        result = torch_rl.reinforce.main.run(
                            env=env, n_games=n_games,
                            gamma=0.99,
                            fc_dims=hidden_layer_size,
                            optimizer_type=NetworkOptimizer.ADAM,
                            optimizer_args=network_optimizer_args,
                            enable_action_blocking=enable_action_blocker,
                            min_penalty=penalty,
                            goal=goal)

                        new_row = {
                            'hidden_layer_size': hidden_layer_size,
                            'learning_rate': learning_rate,
                            'goal_focused': 'Yes' if goal else 'No'
                        }

                        for key in result:
                            new_row.update({key: result[key]})

                        if is_observation_space_well_defined:
                            new_row.update({'normalize_state': 'Yes' if normalize_state else 'No'})

                        results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False)


def run_actor_critic(env, env_name, penalty, env_goal=None):
    n_games = (500, 50)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_actor_critic.csv'.format(env_name))

    result_cols = ['hidden_layer_sizes', 'learning_rate', 'goal_focused',
                   'num_time_steps_train', 'avg_score_train',
                   'action_blocker_precision_train',
                   'action_blocker_recall_train', 'num_actions_blocked_train',
                   'num_time_steps_test', 'avg_score_test',
                   'action_blocker_precision_test', 'action_blocker_recall_test',
                   'num_actions_blocked_test',
                   'avg_actor_loss', 'avg_critic_loss']

    is_observation_space_well_defined = not is_observation_space_not_well_defined(env)

    if is_observation_space_well_defined:
        result_cols.insert(0, 'normalize_states')

    results = pd.DataFrame(columns=result_cols)

    normalize_state_flags = [False, True] if is_observation_space_well_defined else [False]
    enable_action_blocker_flags = [False, True] if penalty > 0 else [False]
    for enable_action_blocker in enable_action_blocker_flags:
        for normalize_state in normalize_state_flags:
            for hidden_layer_size in [64, 128, 256, 512]:
                for learning_rate in [0.001, 0.0003, 5e-6]:
                    for goal in list({None, env_goal}):
                        if normalize_state:
                            env = NormalizedStates(env)
                            if goal:
                                goal = env.observation(goal)
                        network_optimizer_args = {
                            'learning_rate': learning_rate
                        }
                        result = torch_rl.actor_critic.main.run(
                            env=env,
                            n_games=n_games,
                            gamma=0.99,
                            fc_dims=hidden_layer_size,
                            optimizer_type=NetworkOptimizer.ADAM,
                            optimizer_args=network_optimizer_args,
                            enable_action_blocking=enable_action_blocker,
                            min_penalty=penalty,
                            goal=goal)

                        new_row = {
                            'hidden_layer_size': hidden_layer_size,
                            'learning_rate': learning_rate,
                            'goal_focused': 'Yes' if goal else 'No'
                        }
                        for key in result:
                            new_row.update({key: result[key]})
                        if is_observation_space_well_defined:
                            new_row.update({'normalize_state': 'Yes' if normalize_state else 'No'})

                        results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False)


def run_actor_critic_discrete_methods(env, env_name, penalty, env_goal=None):
    print('Running REINFORCE')
    run_reinforce(env, env_name, penalty, env_goal)
    print('Running ActorCritic')
    run_actor_critic(env, env_name, penalty, env_goal)


def run_all_discrete_methods(env, env_name, penalty, env_goal=None):
    print('Running', env_name)
    run_dqn_methods(env, env_name, penalty, env_goal)
    run_actor_critic_discrete_methods(env, env_name, penalty, env_goal)


def run_ppo(env, env_name):
    n_games = (100, 10)

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results', '{0}_ppo.csv'.format(env_name))

    result_cols = ['normalize_actions', 'batch_size', 'hidden_layer_size', 'actor_learning_rate',
                   'critic_learning_rate', 'num_updates_per_iteration', 'num_time_steps_train',
                   'avg_score_train', 'num_time_steps_test',
                   'avg_score_test', 'avg_policy_loss', 'avg_value_loss']
    is_observation_space_well_defined = not is_observation_space_not_well_defined(env)

    if is_observation_space_well_defined:
        result_cols.insert(0, 'normalize_states')

    results = pd.DataFrame(columns=result_cols)

    normalize_state_flags = [False, True] if is_observation_space_well_defined else [False]
    for normalize_state in normalize_state_flags:
        for normalize_actions in [False, True]:
            for batch_size in [2048, 4800]:
                for hidden_layer_size in [64, 128, 256, 512]:
                    for actor_learning_rate in [0.001, 3e-4]:
                        for critic_learning_rate in [0.001, 3e-4]:
                            for num_updates_per_iteration in [5, 10]:
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
                                print('Running instance of PPO: Actions {0}normalized, batch size of {1}, '
                                      'hidden layer size of {2}, actor learning rate of {3}, critic learning rate of '
                                      '{4} and {5} update iterations with {6}normalized states'
                                      .format('' if normalize_actions else 'un', batch_size, hidden_layer_size,
                                              actor_learning_rate, critic_learning_rate, num_updates_per_iteration,
                                              '' if normalize_state else 'un'))
                                num_time_steps_train, avg_score_train, num_time_steps_test, avg_score_test, avg_policy_loss, avg_critic_loss = torch_rl.ppo.main.run(
                                    env=env, n_games=n_games, fc_dims=hidden_layer_size,
                                    actor_optimizer_type=NetworkOptimizer.ADAM,
                                    critic_optimizer_type=NetworkOptimizer.ADAM,
                                    actor_optimizer_args=actor_optimizer_args,
                                    critic_optimizer_args=critic_optimizer_args,
                                    updates_per_iteration=num_updates_per_iteration, batch_size=batch_size)

                                new_row = {
                                    'normalize_actions': 'Yes' if normalize_actions else 'No',
                                    'batch_size': batch_size,
                                    'hidden_layer_size': hidden_layer_size,
                                    'actor_learning_rate': actor_learning_rate,
                                    'critic_learning_rate': critic_learning_rate,
                                    'num_updates_per_iteration': num_updates_per_iteration,
                                    'num_time_steps_train': num_time_steps_train,
                                    'avg_score_train': round(avg_score_train, 5),
                                    'num_time_steps_test': num_time_steps_test,
                                    'avg_score_test': round(avg_score_test, 5),
                                    'avg_policy_loss': round(avg_policy_loss, 5),
                                    'avg_value_loss': round(avg_critic_loss, 5)
                                }

                                if is_observation_space_well_defined:
                                    new_row.update({'normalize_states': 'Yes' if normalize_state else 'No'})

                                results = results.append(new_row, ignore_index=True)

    results.to_csv(csv_file, index=False)


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
                                      derive_hidden_layer_size(env.observation_space.shape, batch_size)]
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

    results.to_csv(csv_file, index=False)


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
                                      derive_hidden_layer_size(env.observation_space.shape, batch_size)]
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

    results.to_csv(csv_file, index=False)


def run_sac(env, env_name):
    n_games = 75

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_sac.csv'.format(env_name))

    result_cols = ['normalize_actions', 'batch_size', 'hidden_layer_size', 'replay',
                   'actor_learning_rate',
                   'critic_learning_rate', 'value_learning_rate', 'tau',
                   'num_time_steps_train', 'avg_score_train',
                   'num_time_steps_test', 'avg_score_test',
                   'avg_policy_loss', 'avg_value_loss']

    is_observation_space_well_defined = not is_observation_space_not_well_defined(env)

    if is_observation_space_well_defined:
        result_cols.insert(0, 'normalize_states')

    results = pd.DataFrame(columns=result_cols)

    normalize_state_flags = [False, True] if is_observation_space_well_defined else [False]
    for normalize_state in normalize_state_flags:
        for normalize_actions in [False, True]:
            for batch_size in [64, 100, 128]:
                hidden_layer_sizes = [64, 128, 256, 300, 400, 512,
                                      derive_hidden_layer_size(env.observation_space.shape, batch_size)]
                hidden_layer_sizes = list(set(hidden_layer_sizes))
                for hidden_layer_size in hidden_layer_sizes:
                    for randomized in [False, True]:
                        for actor_learning_rate in [0.001, 0.0003]:
                            for critic_learning_rate in [0.001, 0.0003]:
                                for value_learning_rate in [0.001, 0.0003]:
                                    for tau in [1e-2, 0.005]:
                                        if normalize_actions:
                                            env = NormalizedActions(env)
                                        actor_optimizer_args = {
                                            'learning_rate': actor_learning_rate
                                        }
                                        critic_optimizer_args = {
                                            'learning_rate': critic_learning_rate
                                        }
                                        value_optimizer_args = {
                                            'learning_rate': value_learning_rate
                                        }
                                        print('Running instance of SAC: Actions {0}normalized, {1} replay, batch size '
                                              'of '
                                              '{2}, '
                                              'hidden layer size of {3}, '
                                              'actor learning rate of {4}, critic learning rate of {5},'
                                              ' value learning rate of {6} and tau {7} with '
                                              '{8}normalized states'
                                              .format('' if normalize_actions else 'un',
                                                      'Randomized' if randomized else 'Sequenced', batch_size,
                                                      hidden_layer_size,
                                                      actor_learning_rate, critic_learning_rate, value_learning_rate,
                                                      tau, '' if normalize_state else 'un'))
                                        num_time_steps_train, avg_score_train, num_time_steps_test, avg_score_test, avg_policy_loss, avg_critic_loss = torch_rl.sac.main.run(
                                            env=env, n_games=n_games, tau=tau, fc_dims=hidden_layer_size,
                                            batch_size=batch_size, randomized=randomized,
                                            actor_optimizer_type=NetworkOptimizer.ADAM,
                                            critic_optimizer_type=NetworkOptimizer.ADAM,
                                            actor_optimizer_args=actor_optimizer_args,
                                            critic_optimizer_args=critic_optimizer_args,
                                            value_optimizer_type=NetworkOptimizer.ADAM,
                                            value_optimizer_args=value_optimizer_args
                                        )

                                        new_row = {
                                            'normalize_actions': 'Yes' if normalize_actions else 'No',
                                            'batch_size': batch_size,
                                            'hidden_layer_size': hidden_layer_size,
                                            'replay': 'Randomized' if randomized else 'Sequenced',
                                            'actor_learning_rate': actor_learning_rate,
                                            'critic_learning_rate': critic_learning_rate,
                                            'value_learning_rate': value_learning_rate,
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

    results.to_csv(csv_file, index=False)


def run_actor_critic_continuous_methods(env, env_name):
    print('Running', env_name)
    run_ppo(env, env_name)
    run_ddpg(env, env_name)
    run_td3(env, env_name)
    run_sac(env, env_name)
