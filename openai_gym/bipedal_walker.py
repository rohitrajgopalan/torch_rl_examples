import gym
import numpy as np

from common.common_gym_observation import run_actor_critic_continuous_methods, run_heuristics

env = gym.make('BipedalWalker-v3')
run_actor_critic_continuous_methods(env, 'bipedal_walker')


def bipdeal_walker_heuristic(self, observation):
    moving_s_base = 4 + 5 * self.moving_leg
    supporting_s_base = 4 + 5 * self.supporting_leg

    hip_targ = [None, None]  # -0.8 .. +1.1
    knee_targ = [None, None]  # -0.6 .. +0.9
    hip_todo = [0.0, 0.0]
    knee_todo = [0.0, 0.0]

    if self.state == 1:  # stay on one leg
        hip_targ[self.moving_leg] = 1.1
        knee_targ[self.moving_leg] = -0.6
        self.supporting_knee_angle += 0.03
        if observation[2] > 0.29:  # Max Speed
            self.supporting_knee_angle += 0.03
        self.supporting_knee_angle = min(self.supporting_knee_angle, self.min_supporting_knee_angle)
        knee_targ[self.supporting_leg] = self.supporting_knee_angle
        if observation[supporting_s_base + 0] < 0.10:  # supporting leg is behind
            self.state = 2
    if self.state == 2:  # Put other down
        hip_targ[self.moving_leg] = 0.1
        knee_targ[self.moving_leg] = self.min_supporting_knee_angle
        knee_targ[self.supporting_leg] = self.supporting_knee_angle
        if observation[moving_s_base + 4]:
            self.state = 3
            self.supporting_knee_angle = min(observation[moving_s_base + 2], self.min_supporting_knee_angle)
    if self.state == 3:  # Push Off
        knee_targ[self.moving_leg] = self.supporting_knee_angle
        knee_targ[self.supporting_leg] = +1.0
        if observation[supporting_s_base + 2] > 0.88 or observation[2] > 1.2 * 0.29:
            self.state = 1
            self.moving_leg = 1 - self.moving_leg
            self.supporting_leg = 1 - self.moving_leg

    if hip_targ[0]:
        hip_todo[0] = 0.9 * (hip_targ[0] - observation[4]) - 0.25 * observation[5]
    if hip_targ[1]:
        hip_todo[1] = 0.9 * (hip_targ[1] - observation[9]) - 0.25 * observation[10]
    if knee_targ[0]:
        knee_todo[0] = 4.0 * (knee_targ[0] - observation[6]) - 0.25 * observation[7]
    if knee_targ[1]:
        knee_todo[1] = 4.0 * (knee_targ[1] - observation[11]) - 0.25 * observation[12]

    hip_todo[0] -= 0.9 * (0 - observation[0]) - 1.5 * observation[1]  # PID to keep head strait
    hip_todo[1] -= 0.9 * (0 - observation[0]) - 1.5 * observation[1]
    knee_todo[0] -= 15.0 * observation[3]  # vertical speed, to damp oscillations
    knee_todo[1] -= 15.0 * observation[3]

    a = np.zeros((4,))

    a[0] = hip_todo[0]
    a[1] = knee_todo[0]
    a[2] = hip_todo[1]
    a[3] = knee_todo[1]
    a = np.clip(0.5 * a, -1.0, 1.0)

    return a


run_heuristics(env, 'bipedal_walker', bipdeal_walker_heuristic, 0, None, state=1, moving_leg=0, supporting_leg=1,
               supporting_knee_angle=0.1, min_supporting_knee_angle=0.1)
