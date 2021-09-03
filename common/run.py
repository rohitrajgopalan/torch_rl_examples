import numpy as np
from gym.spaces import Dict

from petting_zoo.random_legal import RandomLegal

try:
    from common.utils import have_we_ran_out_of_time
except ImportError:
    from common.utils import have_we_ran_out_of_time

from torch_rl.utils.types import LearningType
from torch_rl.heuristic.heuristic_with_dt import HeuristicWithDT
from torch_rl.heuristic.heuristic_with_td import HeuristicWithTD
from torch_rl.heuristic.heuristic_with_td3 import HeuristicWithTD3
from torch_rl.heuristic.heuristic_with_dueling_td import HeuristicWithDuelingTD
from torch_rl.heuristic.heuristic_with_ddpg import HeuristicWithDDPG
from torch_rl.td.agent import TDAgent
from torch_rl.dueling_td.agent import DuelingTDAgent
from torch_rl.ddpg.agent import DDPGAgent
from torch_rl.td3.agent import TD3Agent


def run_gym_env(env, agent, n_games_train, n_games_test, learning_type=LearningType.ONLINE):
    learning_types = [LearningType.OFFLINE, LearningType.ONLINE] if learning_type == LearningType.BOTH else [
        learning_type]

    scores_train = np.zeros((n_games_train * len(learning_types)))
    num_time_steps_train = 0
    for current_learning_type in learning_types:
        for i in range(n_games_train):
            score = 0
            observation = env.reset()
            done = False

            t = 0
            while not done and not have_we_ran_out_of_time(env, t):
                if type(agent) in [HeuristicWithDDPG, HeuristicWithTD3, HeuristicWithTD, HeuristicWithDuelingTD,
                                   HeuristicWithDT]:
                    action = agent.get_action(env, current_learning_type, observation, True, t)
                elif type(agent) in [DDPGAgent, DuelingTDAgent, TDAgent, TD3Agent]:
                    action = agent.choose_action(env, observation, True, t)
                else:
                    action = env.action_space.sample()
                observation_, reward, done, _ = env.step(action)

                if agent.enable_action_blocking and agent.initial_action_blocked and reward > 0:
                    reward *= -1
                score += reward

                agent.store_transition(observation, agent.initial_action,
                                       reward, observation_, done)

                if type(agent) in [DDPGAgent, DuelingTDAgent, TDAgent, TD3Agent]:
                    agent.learn()

                observation = observation_

                t += 1

            scores_train[i] = score
            num_time_steps_train += t
        if type(agent) in [HeuristicWithDDPG, HeuristicWithTD3, HeuristicWithTD, HeuristicWithDuelingTD] or (
                type(agent) == HeuristicWithDT and current_learning_type == LearningType.OFFLINE):
            agent.optimize(env, current_learning_type)

    if n_games_test == 0:
        return {
            'num_time_steps_train': num_time_steps_train,
            'avg_score_train': np.mean(scores_train),
            'num_time_steps_test': 0,
            'avg_score_test': -1
        }
    else:
        scores_test = np.zeros(n_games_test)
        num_time_steps_test = 0

        for i in range(n_games_test):
            score = 0
            observation = env.reset()
            done = False

            t = 0
            while not done and not have_we_ran_out_of_time(env, t):
                if type(agent) in [HeuristicWithDDPG, HeuristicWithTD3, HeuristicWithTD, HeuristicWithDuelingTD,
                                   HeuristicWithDT]:
                    action = agent.get_action(env, learning_type, observation, True, t)
                elif type(agent) in [DDPGAgent, DuelingTDAgent, TDAgent, TD3Agent]:
                    action = agent.choose_action(env, observation, True, t)
                else:
                    action = env.action_space.sample()
                observation_, reward, done, _ = env.step(action)
                score += reward

                t += 1

            scores_test[i] = score
            num_time_steps_test += t

        return {
            'num_time_steps_train': num_time_steps_train,
            'avg_score_train': np.mean(scores_train),
            'num_time_steps_test': num_time_steps_test,
            'avg_score_test': np.mean(scores_test)
        }


def run_pettingzoo_env(env, agents, n_games_train, n_games_test, learning_type=LearningType.ONLINE):
    current_state = {}
    current_action = {}
    cumulative_rewards = {}
    scores_train = {}
    scores_test = {}

    learning_types = [LearningType.OFFLINE, LearningType.ONLINE] if learning_type == LearningType.BOTH else [
        learning_type]

    num_agents = 0

    for agent_id in agents:
        scores_train.update({agent_id: np.zeros((n_games_train * len(learning_types)))})
        scores_test.update({agent_id: np.zeros(n_games_test)})
        num_agents += 1

    n_time_steps_train = 0

    for current_learning_type in learning_types:
        for current_game in range(n_games_train):
            t = 0
            for agent_id in agents:
                current_state.update({agent_id: None})
                current_action.update({agent_id: None})
                cumulative_rewards.update({agent_id: 0})

            for agent_id in env.agent_iter():
                if agent_id in agents:
                    agent = agents[agent_id]
                    past_state = current_state[agent_id]
                    past_action = current_action[agent_id]
                    old_cum_reward = cumulative_rewards[agent_id]

                    state, new_cum_reward, done, _ = env.last()

                    reward = new_cum_reward - old_cum_reward

                    if done:
                        env.step(None)
                    else:
                        if type(agent) in [HeuristicWithDDPG, HeuristicWithTD3, HeuristicWithTD, HeuristicWithDuelingTD,
                                           HeuristicWithDT]:
                            if type(env.observation_spaces[agent_id]) == Dict:
                                state = state['observation']
                            action = agent.get_action(env, current_learning_type, state, True, t)
                        elif type(agent) in [DDPGAgent, DuelingTDAgent, TDAgent, TD3Agent]:
                            if type(env.observation_spaces[agent_id]) == Dict:
                                state = state['observation']
                            action = agent.choose_action(env, state, True, t)
                        else:
                            action = agent.act(state)
                        env.step(action)

                        current_state.update({agent_id: state})
                        current_action.update({agent_id: agent.initial_action})
                        cumulative_rewards.update({agent_id: new_cum_reward})

                        if past_state is not None and past_action is not None and type(agent) != RandomLegal:
                            agent.store_transition(past_state, past_action, reward, state, done)
                            scores_train[agent_id][current_game] += reward

                        if type(agent) in [DDPGAgent, DuelingTDAgent, TDAgent, TD3Agent]:
                            agent.learn()
                else:
                    _, _, done, _ = env.last()
                    env.step(env.action_spaces[agent_id].sample() if not done else None)
                t += 1
            n_time_steps_train += 1
        for agent_id in agents:
            agent = agents[agent_id]
            if type(agent) in [HeuristicWithDDPG, HeuristicWithTD3, HeuristicWithTD, HeuristicWithDuelingTD] or (
                    type(agent) == HeuristicWithDT and current_learning_type == LearningType.OFFLINE):
                agent.optimize(env, current_learning_type)

    avg_reward_train = 0
    for agent_id in agents:
        scores = scores_train[agent_id]
        avg_reward_train += np.mean(scores)

    avg_reward_train /= num_agents

    if n_games_test == 0:
        return {
            'num_time_steps_train': n_time_steps_train,
            'avg_score_train': avg_reward_train,
            'num_time_steps_test': 0,
            'avg_score_test': -1
        }
    else:
        n_time_steps_test = 0

        for current_game in range(n_games_test):
            for agent_id in agents:
                current_state.update({agent_id: None})
                current_action.update({agent_id: None})
                cumulative_rewards.update({agent_id: 0})

            for agent_id in env.agent_iter():
                if agent_id in agents:
                    agent = agents[agent_id]
                    past_state = current_state[agent_id]
                    past_action = current_action[agent_id]
                    old_cum_reward = cumulative_rewards[agent_id]

                    state, new_cum_reward, done, _ = env.last()

                    reward = new_cum_reward - old_cum_reward

                    if done:
                        env.step(None)
                    else:
                        if type(agent) in [HeuristicWithDDPG, HeuristicWithTD3, HeuristicWithTD, HeuristicWithDuelingTD,
                                           HeuristicWithDT]:
                            if type(env.observation_spaces[agent_id]) == Dict:
                                state = state['observation']
                            action = agent.get_action(env, learning_type, state, False)
                        elif type(agent) in [DDPGAgent, DuelingTDAgent, TDAgent, TD3Agent]:
                            if type(env.observation_spaces[agent_id]) == Dict:
                                state = state['observation']
                            action = agent.choose_action(env, state, False)
                        else:
                            action = agent.act(state)
                        env.step(action)

                        current_state.update({agent_id: state})
                        current_action.update({agent_id: agent.initial_action})
                        cumulative_rewards.update({agent_id: new_cum_reward})

                        if past_state is not None and past_action is not None and type(agent) != RandomLegal:
                            scores_test[agent_id][current_game] += reward
                else:
                    _, _, done, _ = env.last()
                    env.step(env.action_spaces[agent_id].sample() if not done else None)
                n_time_steps_test += 1

        avg_reward_test = 0
        for agent_id in agents:
            scores = scores_test[agent_id]
            avg_reward_test += np.mean(scores)

        avg_reward_test /= num_agents

        return {
            'num_time_steps_train': n_time_steps_train,
            'avg_score_train': avg_reward_train,
            'num_time_steps_test': n_time_steps_test,
            'avg_score_test': avg_reward_test
        }