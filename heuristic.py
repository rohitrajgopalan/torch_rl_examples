import gym
import numpy as np

try:
    from common import run_heuristics
except ImportError:
    from .common import run_heuristics

env = gym.make('CartPole-v1')


def cartpole_heuristic(self, observation):
    theta, omega = observation[2], observation[3]
    if abs(theta) < 0.03:
        return 0 if omega < 0 else 1
    else:
        return 0 if theta < 0 else 1


run_heuristics(env, 'cartpole', cartpole_heuristic)

env = gym.make('Blackjack-v0')


def blackjack_heuristic(self, observation):
    total, dealer_card, _ = observation
    if total >= 17 or (13 <= total <= 16 and 2 <= dealer_card <= 6) or (total == 12 and 4 <= dealer_card <= 6):
        return 0
    else:
        return 1


run_heuristics(env, 'blackjack', blackjack_heuristic)

env_discrete = gym.make('LunarLander-v2')


def lunar_lander_discrete_heuristic(self, observation):
    angle_targ = observation[0] * 0.5 + observation[2] * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(
        observation[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - observation[4]) * 0.5 - (observation[5]) * 1.0
    hover_todo = (hover_targ - observation[1]) * 0.5 - (observation[3]) * 0.5

    if observation[6] or observation[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (
                -(observation[3]) * 0.5
        )  # override to reduce fall speed, that's all we need after contact

    if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
        return 2
    elif angle_todo < -0.05:
        return 3
    elif angle_todo > +0.05:
        return 1
    else:
        return 0


run_heuristics(env_discrete, 'lunar_lander_discrete', lunar_lander_discrete_heuristic, 100)

env_continuous = gym.make('LunarLanderContinuous-v2')


def lunar_lander_continuous_heuristic(self, observation):
    angle_targ = observation[0] * 0.5 + observation[2] * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(
        observation[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - observation[4]) * 0.5 - (observation[5]) * 1.0
    hover_todo = (hover_targ - observation[1]) * 0.5 - (observation[3]) * 0.5

    if observation[6] or observation[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (
                -(observation[3]) * 0.5
        )  # override to reduce fall speed, that's all we need after contact

    a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
    return np.clip(a, -1, +1)


run_heuristics(env_continuous, 'lunar_lander_continuous', lunar_lander_continuous_heuristic)

env_discrete = gym.make('MountainCar-v0')


def mountain_car_discrete_heuristic(self, observation):
    position, velocity = observation
    if position < 0:
        return 2 if velocity >= 1e-4 else 0
    else:
        return 0 if velocity >= 1e-4 else 2


run_heuristics(env_discrete, 'mountain_car_discrete', mountain_car_discrete_heuristic, 0, np.array([0.5, 0]))

env_continuous = gym.make('MountainCarContinuous-v0')


def mountain_car_continuous_heuristic(self, observation):
    position, velocity = observation
    if position < 0:
        if velocity >= 1e-4:
            return np.random.uniform(low=0, high=1, size=(1,))
        else:
            return np.random.uniform(low=-1, high=0, size=(1,))
    else:
        if velocity >= 1e-4:
            return np.random.uniform(low=-1, high=0, size=(1,))
        else:
            return np.random.uniform(low=0, high=1, size=(1,))


run_heuristics(env_continuous, 'mountain_car_continuous', mountain_car_continuous_heuristic, 0, np.array([0.45, 0]))

env = gym.make('BipedalWalker-v3')


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

env = gym.make('gym_ccc.envs:Multirotor2DSimpNonNormCont-v0')


def control(self, observation):
    """Compute control given state."""
    gravity = 9.8
    mass = 1

    pos_y_goal = -5
    pos_z_goal = 5
    vel_y_goal = 0
    vel_z_goal = 0
    # roll_ang_goal = 0 implied

    pos_y = observation[0]
    pos_z = observation[1]
    vel_y = observation[2]
    vel_z = observation[3]
    roll_ang = observation[4]

    kp_thrust = 0.005
    kd_thrust = 0.1
    kp_roll_ang_target = 0.1
    kd_roll_ang_target = 2.3
    kp_roll_ang_vel = 1

    thrust_feedforward = gravity * mass
    pos_z_error = pos_z_goal - pos_z
    vel_z_error = vel_z_goal - vel_z
    thrust_feedback = kp_thrust * pos_z_error + kd_thrust * vel_z_error
    thrust = thrust_feedforward + thrust_feedback

    pos_y_error = pos_y_goal - pos_y
    vel_y_error = vel_y_goal - vel_y
    roll_ang_target = kp_roll_ang_target * pos_y_error \
                      + kd_roll_ang_target * vel_y_error
    roll_ang_error = roll_ang_target - roll_ang

    roll_ang_vel = kp_roll_ang_vel * roll_ang_error
    return np.array([roll_ang_vel, thrust])


run_heuristics(env, 'multirotor', control)
