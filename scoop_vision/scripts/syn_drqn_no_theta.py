import sys
import numpy as np
sys.path.append('../..')

from util.utils import LinearSchedule
from util.plot import plotLearningCurve
from agent.syn_agent.syn_drqn_slice_agent import *
from env import ScoopEnv


class DRQN(torch.nn.Module):
    def __init__(self, image_shape, n_actions):
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

        self.lstm = nn.LSTM(img_conv_out_size, 512, batch_first=True)
        self.fc = nn.Linear(512, n_actions)

    def _getImgConvOut(self, shape):
        o = self.img_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, img, hidden=None):
        img = img.float() / 256
        img_shape = img.shape
        img = img.view(img_shape[0]*img_shape[1], img_shape[2], img_shape[3], img_shape[4])
        img_conv_out = self.img_conv(img)
        x = img_conv_out.view(img_shape[0], img_shape[1], -1)
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
        self.state_padding = torch.zeros(self.envs[0].observation_space[0].shape, device=self.device).unsqueeze(0)

    def getStateFromObs(self, obss):
        states = map(lambda x: torch.tensor(x[0], device=self.device, dtype=torch.float).unsqueeze(0)
                     if x is not None else self.state_padding, obss)
        return states


if __name__ == '__main__':
    envs = []
    for i in range(2):
        env = ScoopEnv(19997 + i)
        envs.append(env)

    agent = Agent(DRQN(envs[0].observation_space[0].shape, 4),
                  envs, LinearSchedule(10000, 0.1), batch_size=128, min_mem=200)
    agent.train(10000, 200)
