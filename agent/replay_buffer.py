import numpy as np
import random
from collections import deque, namedtuple
from util.segment_tree import SumSegmentTree, MinSegmentTree

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'final_mask', 'pad_mask'))

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    # def add(self, obs_t, action, reward, obs_tp1, done):
    #     data = (obs_t, action, reward, obs_tp1, done)
    #
    #     if self._next_idx >= len(self._storage):
    #         self._storage.append(data)
    #     else:
    #         self._storage[self._next_idx] = data
    #     self._next_idx = (self._next_idx + 1) % self._maxsize

    def push(self, *args):
        state, action, next_state, reward = args
        if type(state) is tuple:
            state = map(lambda x: x.to('cpu'), state)
        else:
            state = state.to('cpu')

        if next_state is not None:
            final_mask = 0
            if type(next_state) is tuple:
                next_state = map(lambda x: x.to('cpu'), next_state)
            else:
                next_state = next_state.to('cpu')
        else:
            final_mask = 1

        action = action.to('cpu')
        reward = reward.to('cpu')
        data = Transition(state, action, next_state, reward, final_mask, 0)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        samples = []
        for idx in idxes:
            samples.append(self._storage[idx])
        return Transition(*zip(*samples))


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha=0.5):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def push(self, *args):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        ReplayBuffer.push(self, *args)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta=0.5):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)

        samples = []
        for idx in idxes:
            samples.append(self._storage[idx])
        return Transition(*zip(*samples)), weights, idxes

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class EpisodicReplayBuffer:
    def __init__(self, size, sequence_len):
        self.memory = deque()
        self.capacity = size
        self.sequence_len = sequence_len

    def push(self, episode):
        self.memory.append(episode)
        while self.__len__() > self.capacity:
            self.memory.popleft()

    def sample(self, batch_size):
        sample = []
        batch_size = min(batch_size, len(self.memory))
        batch_indexes = np.random.choice(np.arange(len(self.memory)), batch_size, replace=False)
        for batch_idx in batch_indexes:
            episode = self.memory[batch_idx]
            start = random.randint(0, len(episode) - self.sequence_len)
            transitions = episode[start:start + self.sequence_len]
            sample.append(transitions)
        return sample

    def __len__(self):
        return sum(map(len, self.memory))


class PrioritizedEpisodicReplayBuffer:
    def __init__(self, size, sequence_len, alpha):
        self.memory = deque()
        self.capacity = size
        self.sequence_len = sequence_len

        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)

        self._max_priority = 1.0

