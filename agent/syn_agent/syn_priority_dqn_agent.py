from collections import namedtuple, deque
from multiprocessing.pool import ThreadPool as Pool
import random
import time
import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from tqdm import tqdm, trange

import gym

import sys
sys.path.append('../..')
from agent.replay_buffer import *
from util.utils import *
import util.torch_utils as torch_utils
from gym_test.wrapper import wrap_dqn

# Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'final_mask', 'pad_mask'))
#
#
# class ReplayMemory:
#     def __init__(self, capacity):
#         self.memory = deque(maxlen=capacity)
#         self.capacity = capacity
#
#     def push(self, *args):
#         state, action, next_state, reward = args
#         if type(state) is tuple:
#             state = map(lambda x: x.to('cpu'), state)
#         else:
#             state = state.to('cpu')
#
#         if next_state is not None:
#             final_mask = 0
#             if type(next_state) is tuple:
#                 next_state = map(lambda x: x.to('cpu'), next_state)
#             else:
#                 next_state = next_state.to('cpu')
#         else:
#             final_mask = 1
#
#         action = action.to('cpu')
#         reward = reward.to('cpu')
#
#         self.memory.append(Transition(state, action, next_state, reward, final_mask, 0))
#
#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)
#
#     def __len__(self):
#         return len(self.memory)


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

        self.memory = PrioritizedReplayBuffer(memory_size)
        self.criterion = torch_utils.WeightedHuberLoss()
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

        if envs is not None and hasattr(envs[0].observation_space, 'shape'):
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

        mini_batch = memory

        batch_state = map(lambda x: x if x is not None else padding.to('cpu'), mini_batch.state)
        batch_next_state = map(lambda x: x if x is not None else padding.to('cpu'), mini_batch.next_state)

        state = torch.cat(batch_state).to(self.device)
        action = torch.cat(mini_batch.action).to(self.device)
        next_state = torch.cat(batch_next_state).to(self.device)
        reward = torch.cat(mini_batch.reward).to(self.device)
        final_mask = torch.tensor(mini_batch.final_mask, dtype=torch.uint8).to(self.device)
        pad_mask = torch.tensor(mini_batch.pad_mask, dtype=torch.float).to(self.device)
        non_pad_mask = 1 - pad_mask

        return state, action, next_state, reward, final_mask, non_pad_mask

    def optimizeModel(self):
        if len(self.memory) < self.min_mem:
            return
        mini_memory, weights, idxes = self.memory.sample(self.batch_size)

        weights = torch.from_numpy(weights).float().to(self.device)

        state_batch, action_batch, next_state_batch, reward_batch, final_mask, non_pad_mask = self.unzipMemory(mini_memory)

        state_action_values = self.policy_net(state_batch)
        state_action_values = state_action_values.gather(1, action_batch).squeeze(1)
        target_state_action_values = self.target_net(next_state_batch)
        target_state_action_values = target_state_action_values.max(1)[0].detach()

        expected_state_action_values = reward_batch

        target_state_action_values[final_mask] = 0
        expected_state_action_values += self.gamma * target_state_action_values

        loss = self.criterion(state_action_values, expected_state_action_values, weights, non_pad_mask)

        # loss = F.mse_loss(state_action_values[non_pad_mask], expected_state_action_values[non_pad_mask])

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Calc TD error
        with torch.no_grad():
            td_error = (state_action_values - expected_state_action_values) * non_pad_mask
        td_error = td_error.view(self.batch_size, -1).sum(dim=1)
        new_priorities = np.abs(td_error) + 1
        self.memory.update_priorities(idxes, new_priorities)

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

                alive_states = [states[idx] for idx in self.alive_idx]
                alive_actions = [actions[idx] for idx in self.alive_idx]
                if step == max_episode_steps:
                    dones = [True for _ in dones]
                self.pushMemory(alive_states, alive_actions, next_states, rewards, dones)

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
                if len(self.alive_idx) == 0 or step == max_episode_steps:
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
