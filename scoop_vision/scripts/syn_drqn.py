import sys
import numpy as np
sys.path.append('../..')

from util.utils import LinearSchedule
from util.plot import *
from agent.syn_agent.syn_drqn_slice_agent import *
from env import ScoopEnv


class DRQN(torch.nn.Module):
    def __init__(self, image_shape, theta_shape, n_actions):
        super(DRQN, self).__init__()
        self.img_conv = nn.Sequential(
            nn.Conv2d(image_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        img_conv_out_size = self._getImgConvOut(image_shape)

        self.theta_conv = nn.Sequential(
            nn.Conv1d(theta_shape[0], 8, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3),
            nn.ReLU(),
        )

        theta_conv_out_size = self._getThetaConvOut(theta_shape)

        self.lstm = nn.LSTM(img_conv_out_size + theta_conv_out_size, 512, batch_first=True)
        self.fc = nn.Linear(512, n_actions)

    def _getImgConvOut(self, shape):
        o = self.img_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def _getThetaConvOut(self, shape):
        o = self.theta_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, inputs, hidden=None):
        img, theta = inputs
        img_shape = img.shape
        img = img.view(img_shape[0]*img_shape[1], img_shape[2], img_shape[3], img_shape[4])
        img_conv_out = self.img_conv(img)
        img_vec = img_conv_out.view(img_shape[0], img_shape[1], -1)

        theta_shape = theta.shape
        theta = theta.view(theta_shape[0]*theta_shape[1], theta_shape[2], theta_shape[3])
        theta_conv_out = self.theta_conv(theta)
        theta_vec = theta_conv_out.view(img_shape[0], img_shape[1], -1)

        x = torch.cat((img_vec, theta_vec), 2)
        if hidden is None:
            x, hidden = self.lstm(x)
        else:
            x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden


class Agent(SynDRQNAgent):
    def __init__(self, model, envs, exploration,
                 gamma=0.99, memory_size=100000, batch_size=64, target_update_frequency=1000, saving_dir=None,
                 min_mem=1000, sequence_len=10):
        saving_dir = '/home/ur5/thesis/rdd_rl/scoop_vision/data/syn_drqn'
        SynDRQNAgent.__init__(self, model, envs, exploration, gamma, memory_size, batch_size, target_update_frequency,
                              saving_dir, min_mem, sequence_len)
        if envs is not None:
            self.state_padding = (torch.zeros(self.envs[0].observation_space[0].shape, device=self.device).unsqueeze(0),
                              torch.zeros(self.envs[0].observation_space[1].shape, device=self.device).unsqueeze(0))

    def forwardPolicyNet(self, x):
        with torch.no_grad():
            img, theta = x
            img = img.unsqueeze(1)
            theta = theta.unsqueeze(1)
            q_values, self.hidden = self.policy_net((img, theta), self.hidden)
            q_values = q_values.squeeze(1)
            return q_values

    def getStateFromObs(self, obss):
        states = map(lambda x: (torch.tensor(x[0], device=self.device, dtype=torch.float).unsqueeze(0),
                                torch.tensor(x[1], device=self.device, dtype=torch.float).unsqueeze(0))
                     if x is not None else self.state_padding, obss)
        return states

    def getStateInputTensor(self, states):
        imgs, thetas = zip(*states)
        return torch.cat(imgs), torch.cat(thetas)

    def unzipMemory(self, memory):
        state_0_batch = []
        state_1_batch = []

        next_state_0_batch = []
        next_state_1_batch = []

        action_batch = []
        reward_batch = []
        final_mask_batch = []
        pad_mask_batch = []

        state_0_padding = self.state_padding[0].to('cpu')
        state_1_padding = self.state_padding[1].to('cpu')

        for episode in memory:
            episode_transition = Transition(*zip(*episode))
            state_0_batch.append(torch.cat([s[0] if s is not None else state_0_padding
                                            for s in episode_transition.state]))
            state_1_batch.append(torch.cat([s[1] if s is not None else state_1_padding
                                            for s in episode_transition.state]))
            next_state_0_batch.append(torch.cat([s[0] if s is not None else state_0_padding
                                                 for s in episode_transition.next_state]))
            next_state_1_batch.append(torch.cat([s[1] if s is not None else state_1_padding
                                                 for s in episode_transition.next_state]))
            action_batch.append(torch.cat(episode_transition.action))
            reward_batch.append(torch.cat(episode_transition.reward))
            final_mask_batch.append(torch.tensor(list(episode_transition.final_mask), dtype=torch.uint8))
            pad_mask_batch.append(torch.tensor(list(episode_transition.pad_mask), dtype=torch.uint8))

        state = (torch.stack(state_0_batch).to(self.device),
                 torch.stack(state_1_batch).to(self.device))
        action = torch.stack(action_batch).to(self.device)
        next_state = (torch.stack(next_state_0_batch).to(self.device),
                      torch.stack(next_state_1_batch).to(self.device))
        reward = torch.stack(reward_batch).to(self.device)
        final_mask = torch.stack(final_mask_batch).to(self.device)
        pad_mask = torch.stack(pad_mask_batch)
        non_pad_mask = 1 - pad_mask

        return state, action, next_state, reward, final_mask, non_pad_mask


if __name__ == '__main__':
    envs = []
    for i in range(8):
        env = ScoopEnv(19997 + i)
        envs.append(env)

    agent = Agent(DRQN(envs[0].observation_space[0].shape, envs[0].observation_space[1].shape, 4),
                  envs, LinearSchedule(10000, 0.1), batch_size=128, min_mem=1000)
    agent.train(100000, 200, save_freq=500)

