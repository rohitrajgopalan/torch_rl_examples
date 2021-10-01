import numpy as np


class RandomLegal:
    def act(self, observation):
        action_mask = observation['action_mask']
        possible_actions = np.where(action_mask == 1)[0]
        return np.random.choice(possible_actions)
