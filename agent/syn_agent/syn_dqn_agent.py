from collections import namedtuple, deque
from multiprocessing.pool import ThreadPool as Pool
import random
import time
import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm, trange

import gym

import sys
sys.path.append('../..')
from util.utils import *
from gym_test.wrapper import wrap_dqn

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'final_mask', 'pad_mask'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

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

        self.memory.append(Transition(state, action, next_state, reward, final_mask, 0))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class SynDQNAgent:
    def __init__(self, model, envs, exploration,
                 gamma=0.99, memory_size=100000, batch_size=64, target_update_frequency=1000, saving_dir=None, min_mem=1000):

        self.exploration = exploration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        if model:
            self.policy_net = model
            self.target_net = copy.deepcopy(self.policy_net)
            self.policy_net = self.policy_net.to(self.device)
            self.target_net = self.target_net.to(self.device)
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.envs = envs
        self.n_env = len(envs)
        self.alive_idx = [i for i in range(self.n_env)]
        self.pool = Pool(self.n_env)

        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update_frequency
        self.steps_done = 0
        self.episodes_done = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.saving_dir = saving_dir

        self.state = None
        self.min_mem = min_mem
        self.state_padding = None
        if envs is not None:
            self.state_padding = torch.zeros(self.envs[0].observation_space.shape, device=self.device).unsqueeze(0)

    def getAliveEnvs(self):
        return [self.envs[idx] for idx in self.alive_idx]

    def forwardPolicyNet(self, x):
        with torch.no_grad():
            q_values = self.policy_net(x)
            return q_values

    def getStateInputTensor(self, states):
        states_tensor = torch.cat(states)
        return states_tensor

    def selectAction(self, states, require_q=False):
        states_tensor = self.getStateInputTensor(states)
        output = self.forwardPolicyNet(states_tensor)
        actions = []
        q_values = []
        for i in range(len(states)):
            e = self.exploration.value(self.steps_done)
            if random.random() > e:
                action = output[i].max(0)[1].view(1, 1)
            else:
                if hasattr(self.envs[0], 'nA'):
                    action_space = self.envs[0].nA
                else:
                    action_space = self.envs[0].action_space.n
                action = torch.tensor([[random.randrange(action_space)]], device=self.device, dtype=torch.long)
            q_value = output[i].unsqueeze(0).gather(1, action).item()
            actions.append(action)
            q_values.append(q_value)
        if require_q:
            return actions, q_values
        else:
            return actions

    def unzipMemory(self, memory):
        padding = self.state_padding

        mini_batch = Transition(*zip(*memory))

        batch_state = map(lambda x: x if x is not None else padding.to('cpu'), mini_batch.state)
        batch_next_state = map(lambda x: x if x is not None else padding.to('cpu'), mini_batch.next_state)

        state = torch.cat(batch_state).to(self.device)
        action = torch.cat(mini_batch.action).to(self.device)
        next_state = torch.cat(batch_next_state).to(self.device)
        reward = torch.cat(mini_batch.reward).to(self.device)
        final_mask = torch.tensor(mini_batch.final_mask, dtype=torch.uint8).to(self.device)
        pad_mask = torch.tensor(mini_batch.pad_mask, dtype=torch.uint8).to(self.device)
        non_pad_mask = 1 - pad_mask

        return state, action, next_state, reward, final_mask, non_pad_mask

    def optimizeModel(self):
        if len(self.memory) < self.min_mem:
            return
        mini_memory = self.memory.sample(self.batch_size)

        state_batch, action_batch, next_state_batch, reward_batch, final_mask, non_pad_mask = self.unzipMemory(mini_memory)

        state_action_values = self.policy_net(state_batch)
        state_action_values = state_action_values.gather(1, action_batch).squeeze(1)
        target_state_action_values = self.target_net(next_state_batch)
        target_state_action_values = target_state_action_values.max(1)[0].detach()

        expected_state_action_values = reward_batch

        target_state_action_values[final_mask] = 0
        expected_state_action_values += self.gamma * target_state_action_values

        loss = F.mse_loss(state_action_values[non_pad_mask], expected_state_action_values[non_pad_mask])

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    @staticmethod
    def _reset(env):
        return env.reset()

    def getStateFromObs(self, obss):
        states = map(lambda x: torch.tensor(x, device=self.device, dtype=torch.float).unsqueeze(0)
                     if x is not None else self.state_padding, obss)
        return states

    def resetEnv(self):
        obss = self.pool.map(self._reset, self.envs)
        self.alive_idx = [i for i in range(self.n_env)]
        states = self.getStateFromObs(obss)
        return states

    @staticmethod
    def _act(args):
        (env, action) = args
        return env.step(action)

    def takeAction(self, actions):
        alive_envs = []
        alive_actions = []
        for idx in self.alive_idx:
            alive_envs.append(self.envs[idx])
            alive_actions.append(actions[idx])
        # alive_envs = self.getAliveEnvs()
        rets = self.pool.map(self._act, (zip(alive_envs, alive_actions)))
        return rets

    def pushMemory(self, states, actions, next_states, rewards, dones):
        alive_states = []
        alive_actions = []
        for idx in self.alive_idx:
            alive_states.append(states[idx])
            alive_actions.append(actions[idx])
        states, actions = alive_states, alive_actions
        for i in range(len(self.alive_idx)):
            state = states[i]
            action = actions[i]
            next_state = next_states[i]
            reward = rewards[i]
            done = dones[i]
            if done:
                next_state = None
            self.memory.push(state, action, next_state, reward)

    def trainOneEpisode(self, num_episodes, max_episode_steps=100, save_freq=100):
        r_total = [0 for _ in range(self.n_env)]
        states = self.resetEnv()
        with trange(1, max_episode_steps + 1, leave=False) as t:
            for step in t:
                actions, qs = self.selectAction(states, True)
                rets = self.takeAction(map(lambda x: x.item(), actions))
                obs_s, rs, dones, infos = zip(*rets)

                next_states = self.getStateFromObs(obs_s)
                rewards = map(lambda x: torch.tensor([x], device=self.device, dtype=torch.float), rs)
                self.steps_done += len(self.alive_idx)

                self.pushMemory(states, actions, next_states, rewards, dones)

                for i, idx in enumerate(copy.copy(self.alive_idx)):
                    r_total[idx] += rs[i]
                    if dones[i]:
                        self.alive_idx.remove(idx)
                        self.episode_rewards.append(r_total[idx])
                        self.episode_lengths.append(step)
                        self.episodes_done += 1
                        next_states[i] = None

                t.set_postfix_str('step={}, total_reward={}'.format(step, r_total))

                self.optimizeModel()
                if self.steps_done % self.target_update < self.n_env:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                if len(self.alive_idx) == 0 or step == max_episode_steps - 1:
                    tqdm.write('------Episode {} ended, total reward: {}, step: {}------' \
                               .format(self.episodes_done, r_total, step))
                    tqdm.write('------Total steps done: {}, current e: {} ------' \
                               .format(self.steps_done, self.exploration.value(self.steps_done)))
                    if self.episodes_done % save_freq < self.n_env:
                        self.saveCheckpoint()
                    break

                states = filter(lambda x: x is not None, next_states)
                for i in range(self.n_env):
                    if i not in self.alive_idx:
                        states.insert(i, self.state_padding)

    def train(self, num_episodes, max_episode_steps=100, save_freq=100):
        """
        train the network for given number of episodes
        :param num_episodes:
        :param max_episode_steps:
        :param save_freq:
        :return:
        """
        while self.episodes_done < num_episodes:
            self.trainOneEpisode(num_episodes, max_episode_steps, save_freq)
        self.saveCheckpoint()

    def getSavingState(self):
        state = {
            'episode': self.episodes_done,
            'steps': self.steps_done,
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }
        return state

    def saveCheckpoint(self):
        """
        save checkpoint in self.saving_dir
        :return: None
        """
        if self.saving_dir is None:
            return
        time_stamp = time.strftime('%Y%m%d%H%M%S', time.gmtime())
        state_filename = os.path.join(self.saving_dir, 'checkpoint.' + time_stamp + '.pth.tar')
        mem_filename = os.path.join(self.saving_dir, 'memory.' + time_stamp + '.pth.tar')
        state = self.getSavingState()
        memory = {
            'memory': self.memory
        }
        torch.save(state, state_filename)
        torch.save(memory, mem_filename)

    def loadCheckpoint(self, time_stamp, data_only=False, load_memory=True):
        """
        load checkpoint at input time stamp
        :param time_stamp: time stamp for the checkpoint
        :return: None
        """
        state_filename = os.path.join(self.saving_dir, 'checkpoint.' + time_stamp + '.pth.tar')
        mem_filename = os.path.join(self.saving_dir, 'memory.' + time_stamp + '.pth.tar')

        print 'loading checkpoint: ', time_stamp
        checkpoint = torch.load(state_filename)
        if data_only:
            self.episode_rewards = checkpoint['episode_rewards']
            self.episode_lengths = checkpoint['episode_lengths']
            return

        self.episodes_done = checkpoint['episode']
        self.steps_done = checkpoint['steps']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']

        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_net = self.policy_net.to(self.device)
        self.policy_net.train()

        self.target_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net = self.target_net.to(self.device)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if load_memory:
            memory = torch.load(mem_filename)
            self.memory = memory['memory']


class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    envs = []
    for i in range(4):
        env = gym.make("CartPole-v1")
        env.seed(i)
        envs.append(env)

    agent = SynDQNAgent(DQN(), envs, LinearSchedule(10000, 0.02), batch_size=128, min_mem=1000)
    agent.train(10000, 500)


# class DQN(nn.Module):
#     def __init__(self, input_shape, n_actions):
#         super(DQN, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU()
#         )
#         conv_out_size = self._get_conv_out(input_shape)
#
#         self.fc1 = nn.Linear(conv_out_size, 512)
#         self.fc2 = nn.Linear(512, n_actions)
#
#     def _get_conv_out(self, shape):
#         o = self.conv(torch.zeros(1, *shape))
#
#         return int(np.prod(o.size()))
#
#     def forward(self, x):
#         x = x.float() / 256
#         conv_out = self.conv(x)
#         x = conv_out.view(x.size()[0], -1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         return x


# class DQNStackAgent(SynDQNAgent):
#     def __init__(self, *args, **kwargs):
#         SynDQNAgent.__init__(self, *args, **kwargs)
#
#     def resetEnv(self):
#         def reset(env):
#             return env.reset()
#         obss = self.pool.map(reset, self.envs)
#         obss = map(lambda x: np.array(x), obss)
#         self.alive_idx = [i for i in range(self.n_env)]
#         states = map(lambda x: torch.tensor(x, device=self.device, dtype=torch.float).unsqueeze(0), obss)
#         return states
#
#     def getNextState(self, obss):
#         obss = map(lambda x: np.array(x), obss)
#         next_states = map(lambda x: torch.tensor(x, device=self.device, dtype=torch.float).unsqueeze(0)
#                           if x is not None else None, obss)
#         return next_states
#
#
# if __name__ == '__main__':
#     envs = []
#     for i in range(4):
#         env = gym.make('PongNoFrameskip-v4')
#         env = wrap_dqn(env)
#         env.seed(i)
#         envs.append(env)
#
#     agent = DQNStackAgent(DQN(envs[0].observation_space.shape, envs[0].action_space.n), envs, exploration=LinearSchedule(100000, 0.02),
#                           batch_size=128, target_update_frequency=1000, memory_size=100000, min_mem=10000)
#     agent.saving_dir = '/home/ur5/thesis/rdd_rl/gym_test/pong/data/syn_dqn'
#     agent.loadCheckpoint('20190214001100')
#     agent.train(10000, 10000)
