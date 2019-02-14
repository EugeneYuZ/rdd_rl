from collections import namedtuple, deque
import random
import time
import os

import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from syn_dqn_agent import SynDQNAgent

from util.utils import *
from gym_test.wrapper import wrap_drqn


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'final_mask', 'pad_mask'))


class SliceReplayMemory:
    def __init__(self, capacity, sequence_len):
        self.memory = deque()
        self.capacity = capacity
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


class SynDRQNAgent(SynDQNAgent):
    def __init__(self, model, envs, exploration,
                 gamma=0.99, memory_size=100000, batch_size=64, target_update_frequency=1000, saving_dir=None,
                 min_mem=1000, sequence_len=10):
        SynDQNAgent.__init__(self, model, envs, exploration, gamma, memory_size, batch_size, target_update_frequency,
                             saving_dir, min_mem)
        self.memory = SliceReplayMemory(memory_size, sequence_len)
        self.hidden = None
        self.local_memory = [[] for _ in range(self.n_env)]
        self.sequence_len = sequence_len

    def forwardPolicyNet(self, x):
        with torch.no_grad():
            state = x.unsqueeze(1)
            q_values, self.hidden = self.policy_net(state, self.hidden)
            q_values = q_values.squeeze(1)
            return q_values

    def unzipMemory(self, memory):
        state_batch = []
        action_batch = []
        next_state_batch = []
        reward_batch = []
        final_mask_batch = []
        pad_mask_batch = []

        padding = self.state_padding.to('cpu')

        for episode in memory:
            episode_transition = Transition(*zip(*episode))
            state_batch.append(torch.cat([s if s is not None else padding
                                          for s in episode_transition.state]))
            action_batch.append(torch.cat(episode_transition.action))
            next_state_batch.append(torch.cat([s if s is not None else padding
                                               for s in episode_transition.next_state]))
            reward_batch.append(torch.cat(episode_transition.reward))
            final_mask_batch.append(torch.tensor(list(episode_transition.final_mask), dtype=torch.uint8))
            pad_mask_batch.append(torch.tensor(list(episode_transition.pad_mask), dtype=torch.uint8))

        state = torch.stack(state_batch).to(self.device)
        action = torch.stack(action_batch).to(self.device)
        next_state = torch.stack(next_state_batch).to(self.device)
        reward = torch.stack(reward_batch).to(self.device)
        final_mask = torch.stack(final_mask_batch).to(self.device)
        pad_mask = torch.stack(pad_mask_batch)
        non_pad_mask = 1 - pad_mask

        return state, action, next_state, reward, final_mask, non_pad_mask

    def optimizeModel(self):
        if len(self.memory) < self.min_mem:
            return
        mini_memory = self.memory.sample(self.batch_size)

        state_batch, action_batch, next_state_batch, reward_batch, final_mask, non_pad_mask = self.unzipMemory(mini_memory)

        state_action_values, _ = self.policy_net(state_batch)
        state_action_values = state_action_values.gather(2, action_batch).squeeze(2)
        target_state_action_values, _ = self.target_net(next_state_batch)
        target_state_action_values = target_state_action_values.max(2)[0].detach()

        expected_state_action_values = reward_batch

        target_state_action_values[final_mask] = 0
        expected_state_action_values += self.gamma * target_state_action_values

        loss = F.mse_loss(state_action_values[non_pad_mask], expected_state_action_values[non_pad_mask])

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def pushMemory(self, states, actions, next_states, rewards, dones):
        for i, idx in enumerate(self.alive_idx):
            state = states[i].to('cpu')
            action = actions[i].to('cpu')
            next_state = next_states[i].to('cpu')
            reward = rewards[i].to('cpu')
            done = dones[i]
            if not done:
                self.local_memory[idx].append(Transition(state, action, next_state, reward, 0, 0))
            else:
                self.local_memory[idx].append(Transition(state, action, next_state, reward, 1, 0))
                while len(self.local_memory[idx]) < self.sequence_len:
                    self.local_memory[idx].append(Transition(
                        None,
                        torch.tensor([[0]], dtype=torch.long),
                        None,
                        torch.tensor([0.]),
                        0,
                        1
                    ))
                self.memory.push(self.local_memory[idx])
                self.local_memory[idx] = []

    def trainOneEpisode(self, num_episodes, max_episode_steps=100, save_freq=100):
        self.hidden = None
        SynDQNAgent.trainOneEpisode(self, num_episodes, max_episode_steps, save_freq)


# class DRQN(torch.nn.Module):
#     def __init__(self):
#         super(DRQN, self).__init__()
#
#         self.fc1 = nn.Linear(4, 64)
#         self.lstm = nn.LSTM(64, 128, batch_first=True)
#         self.fc2 = nn.Linear(128, 2)
#
#     def forward(self, x, hidden=None):
#         x = F.relu(self.fc1(x))
#         if hidden is None:
#             x, hidden = self.lstm(x)
#         else:
#             x, hidden = self.lstm(x, hidden)
#         x = self.fc2(x)
#         return x, hidden
#
#
# class CartPoleDRQNAgent(SynDRQNAgent):
#     def __init__(self, *args, **kwargs):
#         SynDRQNAgent.__init__(self, *args, **kwargs)
#
#     def takeAction(self, actions):
#         def act(args):
#             (env, action) = args
#             obs, r, done, info = env.step(action)
#             if done:
#                 r = -1
#             return obs, r, done, info
#
#         alive_envs = []
#         alive_actions = []
#         for idx in self.alive_idx:
#             alive_envs.append(self.envs[idx])
#             alive_actions.append(actions[idx])
#         # alive_envs = self.getAliveEnvs()
#         rets = self.pool.map(act, (zip(alive_envs, alive_actions)))
#         return rets
#
#     def train(self, num_episodes, max_episode_steps=100, save_freq=100):
#         while self.episodes_done < num_episodes:
#             self.trainOneEpisode(num_episodes, max_episode_steps, save_freq)
#             if len(self.episode_rewards) > 100:
#                 avg = np.average(self.episode_rewards[-100:])
#                 print 'avg reward in 100 episodes: ', avg
#                 if avg > 195:
#                     print 'solved'
#                     return
#
# if __name__ == '__main__':
#     envs = []
#     for i in range(4):
#         env = gym.make("CartPole-v1")
#         env.seed(i)
#         envs.append(env)
#
#     agent = CartPoleDRQNAgent(DRQN(), envs, LinearSchedule(10000, 0.02), batch_size=128, min_mem=10000, sequence_len=32)
#     agent.train(10000, 500)


class DRQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DRQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)

        self.lstm = nn.LSTM(conv_out_size, 512, batch_first=True)
        self.fc = nn.Linear(512, n_actions)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))

        return int(np.prod(o.size()))

    def forward(self, x, hidden=None):
        x = x.float() / 256
        shape = x.shape
        x = x.view(shape[0]*shape[1], shape[2], shape[3], shape[4])
        conv_out = self.conv(x)
        x = conv_out.view(shape[0], shape[1], -1)
        if hidden is None:
            x, hidden = self.lstm(x)
        else:
            x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden



class DRQNStackAgent(SynDRQNAgent):
    def __init__(self, *args, **kwargs):
        SynDRQNAgent.__init__(self, *args, **kwargs)

    def resetEnv(self):
        def reset(env):
            return env.reset()
        obss = self.pool.map(reset, self.envs)
        obss = map(lambda x: np.array(x), obss)
        self.alive_idx = [i for i in range(self.n_env)]
        states = map(lambda x: torch.tensor(x, device=self.device, dtype=torch.float).unsqueeze(0), obss)
        return states

    def getNextState(self, obss):
        obss = map(lambda x: np.array(x), obss)
        next_states = map(lambda x: torch.tensor(x, device=self.device, dtype=torch.float).unsqueeze(0)
                          if x is not None else None, obss)
        return next_states


if __name__ == '__main__':
    envs = []
    for i in range(4):
        env = gym.make('PongNoFrameskip-v4')
        env = wrap_drqn(env)
        env.seed(i)
        envs.append(env)

    agent = DRQNStackAgent(DRQN(envs[0].observation_space.shape, envs[0].action_space.n), envs, exploration=LinearSchedule(100000, 0.02),
                          batch_size=128, target_update_frequency=1000, memory_size=100000, min_mem=10000, sequence_len=10)
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/gym_test/pong/data/syn_drqn'
    agent.train(10000, 10000, 200)