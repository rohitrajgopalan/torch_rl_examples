import os

import pandas as pd

import numpy as np
import gym

import torch_rl.dqn.main
import torch_rl.ddqn.main
import torch_rl.dueling_dqn.main
import torch_rl.dueling_ddqn.main
import torch_rl.reinforce.main
import torch_rl.actor_critic.main
import torch_rl.ddpg.main
import torch_rl.td3.main
import torch_rl.sac.main
from torch_rl.utils.types import NetworkOptimizer


def derive_hidden_layer_size(state_dim, batch_size):
    return state_dim[0] * batch_size


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


def run_dqn(env, env_name):
    n_games = 500

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results', '{0}_dqn.csv'.format(env_name))

    results = pd.DataFrame(columns=['batch_size', 'hidden_layer_size', 'replay', 'optimizer', 'learning_rate',
                                    'num_time_steps', 'avg_score', 'avg_loss'])

    for batch_size in [32, 64, 128]:
        for randomized in [False, True]:
            for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                for learning_rate in [0.001, 0.0001]:
                    hidden_layer_sizes = [derive_hidden_layer_size(env.observation_space.shape, batch_size),
                                          64, 128, 256, 512]
                    hidden_layer_sizes = list(hidden_layer_sizes)
                    for hidden_layer_size in hidden_layer_sizes:
                        network_optimizer_args = {
                            'learning_rate': learning_rate
                        }
                        print('Running Instance of DQN: {0} replay, {1} optimizer with learning rate {2}, '
                              'Batch size of {3} and hidden layer size of {4}'
                              .format('Randomized' if randomized else 'Sequenced', optimizer_type.name.lower(),
                                      learning_rate, batch_size, hidden_layer_size))
                        num_timesteps, avg_score, avg_loss = torch_rl.dqn.main.run(env=env, n_games=n_games, gamma=0.99,
                                                                                   epsilon=1.0, mem_size=1000,
                                                                                   batch_size=batch_size,
                                                                                   fc_dims=hidden_layer_size,
                                                                                   optimizer_type=optimizer_type,
                                                                                   eps_min=0.01,
                                                                                   eps_dec=5e-7,
                                                                                   replace=1000,
                                                                                   optimizer_args=network_optimizer_args,
                                                                                   randomized=randomized)

                        results = results.append({
                            'batch_size': batch_size,
                            'hidden_layer_size': hidden_layer_size,
                            'replay': 'Randomized' if randomized else 'Sequenced',
                            'optimizer': optimizer_type.name.lower(),
                            'learning_rate': learning_rate,
                            'num_time_steps': num_timesteps,
                            'avg_score': round(avg_score, 5),
                            'avg_loss': round(avg_loss, 5)
                        }, ignore_index=True)

    results.to_csv(csv_file, index=False)


def run_ddqn(env, env_name):
    n_games = 500

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results', '{0}_ddqn.csv'.format(env_name))

    results = pd.DataFrame(columns=['batch_size', 'hidden_layer_size', 'replay', 'optimizer', 'learning_rate',
                                    'num_time_steps', 'avg_score', 'avg_loss'])

    for batch_size in [32, 64, 128]:
        for randomized in [False, True]:
            for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                for learning_rate in [0.001, 0.0001]:
                    hidden_layer_sizes = [derive_hidden_layer_size(env.observation_space.shape, batch_size),
                                          64, 128, 256, 512]
                    hidden_layer_sizes = list(hidden_layer_sizes)
                    for hidden_layer_size in hidden_layer_sizes:
                        network_optimizer_args = {
                            'learning_rate': learning_rate
                        }
                        print('Running Instance of DDQN: {0} replay, {1} optimizer with learning rate {2}, '
                              'Batch size of {3} and hidden layer size of {4}'
                              .format('Randomized' if randomized else 'Sequenced', optimizer_type.name.lower(),
                                      learning_rate, batch_size, hidden_layer_size))
                        num_timesteps, avg_score, avg_loss = torch_rl.dqn.main.run(env=env, n_games=n_games, gamma=0.99,
                                                                                   epsilon=1.0, mem_size=1000,
                                                                                   batch_size=batch_size,
                                                                                   fc_dims=hidden_layer_size,
                                                                                   optimizer_type=optimizer_type,
                                                                                   eps_min=0.01,
                                                                                   eps_dec=5e-7,
                                                                                   replace=1000,
                                                                                   optimizer_args=network_optimizer_args,
                                                                                   randomized=randomized)

                        results = results.append({
                            'batch_size': batch_size,
                            'hidden_layer_size': hidden_layer_size,
                            'replay': 'Randomized' if randomized else 'Sequenced',
                            'optimizer': optimizer_type.name.lower(),
                            'learning_rate': learning_rate,
                            'num_time_steps': num_timesteps,
                            'avg_score': round(avg_score, 5),
                            'avg_loss': round(avg_loss, 5)
                        }, ignore_index=True)

    results.to_csv(csv_file, index=False)


def run_dueling_dqn(env, env_name):
    n_games = 500

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_dqn.csv'.format(env_name))

    results = pd.DataFrame(columns=['batch_size', 'hidden_layer_size', 'replay', 'optimizer', 'learning_rate',
                                    'num_time_steps', 'avg_score', 'avg_loss'])

    for batch_size in [32, 64, 128]:
        for randomized in [False, True]:
            for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                for learning_rate in [0.001, 0.0001]:
                    hidden_layer_sizes = [derive_hidden_layer_size(env.observation_space.shape, batch_size),
                                          64, 128, 256, 512]
                    hidden_layer_sizes = list(hidden_layer_sizes)
                    for hidden_layer_size in hidden_layer_sizes:
                        network_optimizer_args = {
                            'learning_rate': learning_rate
                        }
                        print('Running Instance of Dueling DQN: {0} replay, {1} optimizer with learning rate {2}, '
                              'Batch size of {3} and hidden layer size of {4}'
                              .format('Randomized' if randomized else 'Sequenced', optimizer_type.name.lower(),
                                      learning_rate, batch_size, hidden_layer_size))
                        num_timesteps, avg_score, avg_loss = torch_rl.dueling_dqn.main.run(env=env, n_games=n_games,
                                                                                           gamma=0.99,
                                                                                           epsilon=1.0, mem_size=1000,
                                                                                           batch_size=batch_size,
                                                                                           fc_dims=hidden_layer_size,
                                                                                           optimizer_type=optimizer_type,
                                                                                           eps_min=0.01,
                                                                                           eps_dec=5e-7,
                                                                                           replace=1000,
                                                                                           optimizer_args=network_optimizer_args,
                                                                                           randomized=randomized)

                        results = results.append({
                            'batch_size': batch_size,
                            'hidden_layer_size': hidden_layer_size,
                            'replay': 'Randomized' if randomized else 'Sequenced',
                            'optimizer': optimizer_type.name.lower(),
                            'learning_rate': learning_rate,
                            'num_time_steps': num_timesteps,
                            'avg_score': round(avg_score, 5),
                            'avg_loss': round(avg_loss, 5)
                        }, ignore_index=True)

    results.to_csv(csv_file, index=False)


def run_dueling_ddqn(env, env_name):
    n_games = 500

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_dueling_ddqn.csv'.format(env_name))

    results = pd.DataFrame(columns=['batch_size', 'hidden_layer_size', 'replay', 'optimizer', 'learning_rate',
                                    'num_time_steps', 'avg_score', 'avg_loss'])

    for batch_size in [32, 64, 128]:
        for randomized in [False, True]:
            for optimizer_type in [NetworkOptimizer.ADAM, NetworkOptimizer.RMSPROP]:
                for learning_rate in [0.001, 0.0001]:
                    hidden_layer_sizes = [derive_hidden_layer_size(env.observation_space.shape, batch_size),
                                          64, 128, 256, 512]
                    hidden_layer_sizes = list(hidden_layer_sizes)
                    for hidden_layer_size in hidden_layer_sizes:
                        network_optimizer_args = {
                            'learning_rate': learning_rate
                        }
                        print('Running Instance of Dueling DDQN: {0} replay, {1} optimizer with learning rate {2}, '
                              'Batch size of {3} and hidden layer size of {4}'
                              .format('Randomized' if randomized else 'Sequenced', optimizer_type.name.lower(),
                                      learning_rate, batch_size, hidden_layer_size))
                        num_timesteps, avg_score, avg_loss = torch_rl.dueling_ddqn.main.run(env=env, n_games=n_games,
                                                                                            gamma=0.99,
                                                                                            epsilon=1.0, mem_size=1000,
                                                                                            batch_size=batch_size,
                                                                                            fc_dims=hidden_layer_size,
                                                                                            optimizer_type=optimizer_type,
                                                                                            eps_min=0.01,
                                                                                            eps_dec=5e-7,
                                                                                            replace=1000,
                                                                                            optimizer_args=network_optimizer_args,
                                                                                            randomized=randomized)

                        results = results.append({
                            'batch_size': batch_size,
                            'hidden_layer_size': hidden_layer_size,
                            'replay': 'Randomized' if randomized else 'Sequenced',
                            'optimizer': optimizer_type.name.lower(),
                            'learning_rate': learning_rate,
                            'num_time_steps': num_timesteps,
                            'avg_score': round(avg_score, 5),
                            'avg_loss': round(avg_loss, 5)
                        }, ignore_index=True)

    results.to_csv(csv_file, index=False)


def run_dqn_methods(env, env_name):
    run_dqn(env, env_name)
    run_ddqn(env, env_name)
    run_dueling_dqn(env, env_name)
    run_dueling_ddqn(env, env_name)


def run_reinforce(env, env_name):
    n_games = 500

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_reinforce.csv'.format(env_name))

    results = pd.DataFrame(columns=['hidden_layer_size', 'learning_rate', 'num_time_steps', 'avg_score', 'avg_loss'])

    for hidden_layer_size in [64, 128, 256, 512]:
        for learning_rate in [0.001, 0.0005]:
            network_optimizer_args = {
                'learning_rate': learning_rate
            }
            print('Running Instance of REINFORCE with learning rate {0} and hidden layer size of {1}'
                  .format(learning_rate, hidden_layer_size))
            num_timesteps, avg_score, avg_loss = torch_rl.reinforce.main.run(env=env, n_games=n_games,
                                                                             gamma=0.99,
                                                                             fc_dims=hidden_layer_size,
                                                                             optimizer_type=NetworkOptimizer.ADAM,
                                                                             optimizer_args=network_optimizer_args)

            results = results.append({
                'hidden_layer_size': hidden_layer_size,
                'learning_rate': learning_rate,
                'num_time_steps': num_timesteps,
                'avg_score': round(avg_score, 5),
                'avg_loss': round(avg_loss, 5)
            }, ignore_index=True)

    results.to_csv(csv_file, index=False)


def run_actor_critic(env, env_name):
    n_games = 500

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_actor_critic.csv'.format(env_name))

    results = pd.DataFrame(columns=['learning_rate', 'num_time_steps', 'avg_score',
                                    'avg_actor_loss', 'avg_critic_loss'])

    for hidden_layer_size in [64, 128, 256, 512]:
        for learning_rate in [0.001, 0.0003, 5e-6]:
            network_optimizer_args = {
                'learning_rate': learning_rate
            }
            print('Running Instance of Actor Critic with learning rate {0} and hidden layer size of {1}'
                  .format(learning_rate, hidden_layer_size))
            num_timesteps, avg_score, avg_actor_loss, avg_critic_loss = torch_rl.actor_critic.main.run(env=env,
                                                                                                       n_games=n_games,
                                                                                                       gamma=0.99,
                                                                                                       fc_dims=hidden_layer_size,
                                                                                                       optimizer_type=NetworkOptimizer.ADAM,
                                                                                                       optimizer_args=network_optimizer_args)

            results = results.append({
                'hidden_layer_size': hidden_layer_size,
                'learning_rate': learning_rate,
                'num_time_steps': int(num_timesteps),
                'avg_score': round(avg_score, 5),
                'avg_actor_loss': round(avg_actor_loss, 5),
                'avg_critic_loss': round(avg_critic_loss, 5)
            }, ignore_index=True)

    results.to_csv(csv_file, index=False)


def run_actor_critic_discrete_methods(env, env_name):
    run_reinforce(env, env_name)
    run_actor_critic(env, env_name)


def run_all_discrete_methods(env, env_name):
    print('Running', env_name)
    run_dqn_methods(env, env_name)
    run_actor_critic_discrete_methods(env, env_name)


def run_ddpg(env, env_name):
    n_games = 75

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_ddpg.csv'.format(env_name))

    results = pd.DataFrame(columns=['normalize_actions','batch_size', 'hidden_layer_size', 'replay', 'actor_learning_rate',
                                    'critic_learning_rate', 'tau',
                                    'num_time_steps', 'avg_score', 'avg_policy_loss', 'avg_value_loss'])

    for normalize_actions in [False, True]:
        for batch_size in [64, 128]:
            for hidden_layer_size in [64, 128, 256, 300, 400, 512, derive_hidden_layer_size(env.observation_space.shape, batch_size)]:
                for randomized in [False, True]:
                    for actor_learning_rate in [0.001, 0.0001]:
                        for critic_learning_rate in [0.001, 0.0001]:
                            for tau in [1e-2, 1e-3]:
                                if normalize_actions:
                                    env = NormalizedActions(env)
                                actor_optimizer_args = {
                                    'learning_rate': actor_learning_rate
                                }
                                critic_optimizer_args = {
                                    'learning_rate': critic_learning_rate
                                }
                                print('Running instance of DDPG: Actions {0}normalized, {1} replay, batch size of {2}, '
                                      'hidden layer size of {3}, '
                                      'actor learning rate of {4}, critic learning rate of {5} and tau {6}'
                                      .format('' if normalize_actions else 'un', 'Randomized' if randomized else 'Sequenced', batch_size, hidden_layer_size,
                                              actor_learning_rate, critic_learning_rate, tau))
                                num_time_steps, avg_score, avg_policy_loss, avg_critic_loss = torch_rl.ddpg.main.run(
                                    env=env, n_games=n_games, tau=tau, fc_dims=hidden_layer_size,
                                    batch_size=batch_size, randomized=randomized,
                                    actor_optimizer_type=NetworkOptimizer.ADAM,
                                    critic_optimizer_type=NetworkOptimizer.ADAM, actor_optimizer_args=actor_optimizer_args,
                                    critic_optimizer_args=critic_optimizer_args
                                )

                                results = results.append({
                                    'normalize_actions': 'Yes' if normalize_actions else 'No',
                                    'batch_size': batch_size,
                                    'hidden_layer_size': hidden_layer_size,
                                    'replay': 'Randomized' if randomized else 'Sequenced',
                                    'actor_learning_rate': actor_learning_rate,
                                    'critic_learning_rate': critic_learning_rate,
                                    'tau': tau,
                                    'num_time_steps': num_time_steps,
                                    'avg_score': round(avg_score, 5),
                                    'avg_policy_loss': round(avg_policy_loss, 5),
                                    'avg_value_loss': round(avg_critic_loss, 5)
                                }, ignore_index=True)

    results.to_csv(csv_file, index=False)


def run_td3(env, env_name):
    n_games = 75

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_td3.csv'.format(env_name))

    results = pd.DataFrame(columns=['normalize_actions','batch_size', 'hidden_layer_size', 'replay', 'actor_learning_rate',
                                    'critic_learning_rate', 'tau',
                                    'num_time_steps', 'avg_score', 'avg_policy_loss', 'avg_value1_loss',
                                    'avg_value2_loss'])

    actor_optimizer_args = {
        'learning_rate': 1e-3
    }
    critic_optimizer_args = {
        'learning_rate': 1e-3
    }

    for normalize_actions in [False, True]:
        for batch_size in [64, 100, 128]:
            for hidden_layer_size in [64, 128, 256, 300, 400, 512, derive_hidden_layer_size(env.observation_space.shape, batch_size)]:
                for randomized in [False, True]:
                    for tau in [0.005, 0.01]:
                        if normalize_actions:
                            env = NormalizedActions(env)
                        print('Running instance of TD3: Actions {0}normalized, {1} replay, batch size of {2}, '
                              'hidden layer size of {3}, '
                              'and tau {4}'
                              .format('' if normalize_actions else 'un', 'Randomized' if randomized else 'Sequenced', batch_size, hidden_layer_size,
                                      tau))
                        num_time_steps, avg_score, avg_policy_loss, avg_value1_loss, avg_value2_loss = torch_rl.td3.main.run(
                            env=env, n_games=n_games, tau=tau, fc_dims=hidden_layer_size,
                            batch_size=batch_size, randomized=randomized,
                            actor_optimizer_type=NetworkOptimizer.ADAM,
                            critic_optimizer_type=NetworkOptimizer.ADAM, actor_optimizer_args=actor_optimizer_args,
                            critic_optimizer_args=critic_optimizer_args
                        )

                        results = results.append({
                            'normalize_actions': normalize_actions,
                            'batch_size': batch_size,
                            'hidden_layer_size': hidden_layer_size,
                            'replay': 'Randomized' if randomized else 'Sequenced',
                            'tau': tau,
                            'num_time_steps': num_time_steps,
                            'avg_score': round(avg_score, 5),
                            'avg_policy_loss': round(avg_policy_loss, 5),
                            'avg_value1_loss': round(avg_value1_loss, 5),
                            'avg_value2_loss': round(avg_value2_loss, 5)
                        }, ignore_index=True)

    results.to_csv(csv_file, index=False)


def run_sac(env, env_name):
    n_games = 75

    csv_file = os.path.join(os.path.realpath(os.path.dirname('__file__')), 'results',
                            '{0}_sac.csv'.format(env_name))

    results = pd.DataFrame(columns=['normalize_actions','batch_size', 'hidden_layer_size', 'replay',
                                    'actor_learning_rate',
                                    'critic_learning_rate', 'value_learning_rate', 'tau',
                                    'num_time_steps', 'avg_score', 'avg_policy_loss', 'avg_value_loss'])

    for normalize_actions in [False, True]:
        for batch_size in [64, 100, 128]:
            for hidden_layer_size in [64, 128, 256, 512, derive_hidden_layer_size(env.observation_space.shape, batch_size)]:
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
                                    print('Running instance of SAC: Actions {0}normalized, {1} replay, batch size of '
                                          '{2}, '
                                          'hidden layer size of {3}, '
                                          'actor learning rate of {4}, critic learning rate of {5},'
                                          ' value learning rate of {6} and tau {7}'
                                          .format('' if normalize_actions else 'un', 'Randomized' if randomized else 'Sequenced', batch_size, hidden_layer_size,
                                                  actor_learning_rate, critic_learning_rate, value_learning_rate, tau))
                                    num_time_steps, avg_score, avg_policy_loss, avg_critic_loss = torch_rl.sac.main.run(
                                        env=env, n_games=n_games, tau=tau, fc_dims=hidden_layer_size,
                                        batch_size=batch_size, randomized=randomized,
                                        actor_optimizer_type=NetworkOptimizer.ADAM,
                                        critic_optimizer_type=NetworkOptimizer.ADAM, actor_optimizer_args=actor_optimizer_args,
                                        critic_optimizer_args=critic_optimizer_args,
                                        value_optimizer_type=NetworkOptimizer.ADAM, value_optimizer_args=value_optimizer_args
                                    )

                                    results = results.append({
                                        'normalize_actions': 'Yes' if normalize_actions else 'No',
                                        'batch_size': batch_size,
                                        'hidden_layer_size': hidden_layer_size,
                                        'replay': 'Randomized' if randomized else 'Sequenced',
                                        'actor_learning_rate': actor_learning_rate,
                                        'critic_learning_rate': critic_learning_rate,
                                        'value_learning_rate': value_learning_rate,
                                        'tau': tau,
                                        'num_time_steps': num_time_steps,
                                        'avg_score': round(avg_score, 5),
                                        'avg_policy_loss': round(avg_policy_loss, 5),
                                        'avg_value_loss': round(avg_critic_loss, 5)
                                    }, ignore_index=True)

    results.to_csv(csv_file, index=False)


def run_actor_critic_continuous_methods(env, env_name):
    print('Running', env_name)
    run_ddpg(env, env_name)
    run_td3(env, env_name)
    run_sac(env, env_name)
