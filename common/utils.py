from torch_rl.replay.replay import ReplayBuffer
from gym.spaces import Dict, Box


def develop_memory_for_dt_action_blocker_for_gym_env(env, max_time_steps=1000000):
    memory = ReplayBuffer(max_time_steps, env.observation_space.shape)

    obs = env.reset()
    for _ in range(max_time_steps):
        action = env.action_space.sample()
        obs_, reward, done, _ = env.step(action)

        memory.store_transition(obs, action, reward, obs_, done)

        if done:
            env.reset()

    return memory


def develop_memory_for_dt_action_blocker_for_pettingzoo_env(env, max_time_steps=20000000, agent_id=None):
    if agent_id is None:
        agent_id = env.possible_agents[0]

    observation_space = env.observation_spaces[agent_id]

    if type(observation_space) == Dict:
        memory = ReplayBuffer(max_time_steps, observation_space['observation'].shape)
    else:
        memory = ReplayBuffer(max_time_steps, observation_space.shape)

    t = 0
    while t < max_time_steps:
        env.reset()
        current_cum_reward = 0
        current_state = None
        current_action = None
        for agent in env.agent_iter():
            if agent == agent_id:
                old_cum_reward = current_cum_reward
                old_state = current_state
                old_action = current_action

                if type(observation_space) == Dict:
                    obs, current_cum_reward, done, _ = env.last()
                    current_state = obs['observation']
                else:
                    current_state, current_cum_reward, done, _ = env.last()

                if old_state is not None and old_action is not None:
                    reward = current_cum_reward - old_cum_reward
                    memory.store_transition(old_state, old_action, reward, current_state, done)

                current_action = env.action_spaces[agent].sample() if not done else None
                env.step(current_action)
                t += 1
                if t == max_time_steps:
                    break
            else:
                _, _, done, _ = env.last()
                env.step(env.action_spaces[agent].sample() if not done else None)

    return memory


def have_we_ran_out_of_time(env, current_t):
    return hasattr(env, '_max_episode_steps') and current_t == env._max_episode_steps


def generate_agents_for_petting_zoo(env, agent, agent_ids=[]):
    if len(agent_ids) == 0:
        agent_ids = env.possible_agents

    agents = {}

    for agent_id in agent_ids:
        if agent_id in env.possible_agents:
            agents.update({agent_id: agent})

    return agents


def derive_hidden_layer_size(env, batch_size):
    if type(env.observation_space) == Box:
        return env.observation_space.shape[0] * batch_size
    else:
        return batch_size