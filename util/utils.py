import numpy as np


def greedyTieBreak(q_curr):
    max_q = np.max(q_curr)
    candidate = []
    for i, q_value in enumerate(q_curr):
        if q_value == max_q:
            candidate.append(i)
    return np.random.choice(candidate)


def eGreedyActionSelection(q_curr, eps):
    '''
    Preforms epsilon greedy action selectoin based on the Q-values.

    Args:
        q_curr: A numpy array that contains the Q-values for each action for a state.
        eps: The probability to select a random action. Float between 0 and 1.

    Returns:
        The selected action.
    '''

    if np.random.random() < eps:
        return np.random.randint(len(q_curr))
    else:
        return greedyTieBreak(q_curr)


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        '''
        Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Args:
            - schedule_timesteps: Number of timesteps for which to linearly anneal initial_p to final_p
            - initial_p: initial output value
            -final_p: final output value
        '''
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

