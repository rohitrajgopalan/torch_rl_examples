import numpy as np


class TicTacToeHeuristic:
    def __init__(self, my_player_id):
        self.my_player_id = my_player_id
        self.actions_can_be_taken = {
            (0, 1): 2, (0, 2): 1, (0, 3): 6, (0, 4): 8, (0, 6): 3, (0, 8): 4,
            (1, 0): 2, (1, 2): 0, (1, 4): 7, (1, 7): 4,
            (2, 0): 1, (2, 1): 0, (2, 4): 6, (2, 5): 8, (2, 6): 4, (2, 8): 5,
            (3, 0): 6, (3, 4): 5, (3, 5): 4, (3, 6): 0,
            (4, 0): 8, (4, 1): 7, (4, 2): 6, (4, 5): 3, (4, 6): 2, (4, 7): 1, (4, 8): 0,
            (5, 2): 8, (5, 3): 4, (5, 4): 3, (5, 8): 2,
            (6, 0): 3, (6, 2): 4, (6, 3): 0, (6, 4): 2, (6, 7): 8, (6, 8): 7,
            (7, 1): 4, (7, 4): 1, (7, 6): 8, (7, 8): 6,
            (8, 0): 4, (8, 2): 5, (8, 4): 0, (8, 5): 2, (8, 6): 7, (8, 7): 6
        }

    def get_tactical_action(self, legal_actions, actions_already_taken):
        if len(actions_already_taken) == 0:
            return None

        possible_actions = []
        for i in actions_already_taken:
            for j in actions_already_taken:
                t = (i, j)
                if t in self.actions_can_be_taken:
                    possible_actions.append(self.actions_can_be_taken[t])
        if len(possible_actions) == 0:
            return None
        else:
            available_actions = np.intersect1d(np.array(possible_actions), legal_actions)
            return available_actions[0] if available_actions.shape[0] > 0 else None

    def act(self, observation):
        my_actions_taken = []
        other_actions_taken = []

        state = observation['observation']
        legal_actions = np.where(observation['action_mask'] == 1)[0]

        for player_id in range(2):
            player_grid = state[:, :, player_id]
            where = np.where(player_grid == 1)
            x, y = where
            for i in range(x.shape[0]):
                action_taken = 3*x[i] + y[i]
                if player_id == 0:
                    my_actions_taken.append(action_taken)
                else:
                    other_actions_taken.append(action_taken)

        action_to_stop_opponent = self.get_tactical_action(legal_actions, other_actions_taken)
        action_to_win_game = self.get_tactical_action(legal_actions, my_actions_taken)

        if action_to_stop_opponent:
            return action_to_stop_opponent
        elif action_to_win_game:
            return action_to_win_game
        else:
            if 4 in legal_actions:
                return 4
            elif 0 in legal_actions:
                return 0
            elif 2 in legal_actions:
                return 2
            elif 6 in legal_actions:
                return 6
            elif 8 in legal_actions:
                return 8
            else:
                return np.random.choice(legal_actions)


