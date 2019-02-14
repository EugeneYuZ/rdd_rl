import sys
import numpy as np
sys.path.append('../..')

from util.utils import LinearSchedule
from util.plot import plotLearningCurve
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
        img_conv_out_size = self._get_conv_out(image_shape)

        self.theta_conv = nn.Sequential(
            nn.Conv1d(theta_shape[0], 8, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3),
            nn.ReLU(),
        )

        theta_conv_out_size = self._get_conv_out(theta_shape)

        self.lstm = nn.LSTM(img_conv_out_size + theta_conv_out_size, 512, batch_first=True)
        self.fc = nn.Linear(512, n_actions)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))

        return int(np.prod(o.size()))

    def forward(self, inputs, hidden=None):
        img, theta = inputs
        img = img.float() / 256
        img_shape = img.shape
        img = img.view(img_shape[0]*img_shape[1], img_shape[2], img_shape[3], img_shape[4])
        img_conv_out = self.img_conv(img)
        img_vec = img_conv_out.view(img_shape[0], img_shape[1], -1)

        theta_shape = theta.shape
        theta = theta.view(theta_shape[0]*theta_shape[1], theta_shape[2], theta_shape[3])
        theta_conv_out = self.theta_conv(theta)
        theta_vec = theta_conv_out.view(img_shape[0], img_shape[1], -1)

        x = torch.cat((img_vec, theta_vec, 2))
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

    def getStateFromObs(self, obss):
        states = map(lambda x: (torch.tensor(x[0], device=self.device, dtype=torch.float).unsqueeze(0),
                                torch.tensor(x[1], device=self.device, dtype=torch.float).unsqueeze(0)), obss)
        return states

    def getStateInputTensor(self, states):
        imgs, thetas = zip(*states)
        return torch.cat(imgs), torch.cat(thetas)

    def pushMemory(self, states, actions, next_states, rewards, dones):
        for i, idx in enumerate(self.alive_idx):
            state = (states[i][0].to('cpu'), states[i][1].to('cpu'))
            action = actions[i].to('cpu')
            next_state = (next_state[i][0].to('cpu'), next_state[i][1].to('cpu'))
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



