import sys
import numpy as np
sys.path.append('../..')

from util.utils import LinearSchedule
from util.plot import *
from agent.syn_agent.syn_dqn_agent import *
from env_dense_r import ScoopEnv


class DQN(torch.nn.Module):
    def __init__(self, image_shape, theta_shape, n_actions):
        super(DQN, self).__init__()
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

        self.fc1 = nn.Linear(img_conv_out_size + theta_conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def _getImgConvOut(self, shape):
        o = self.img_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def _getThetaConvOut(self, shape):
        o = self.theta_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, inputs):
        img, theta = inputs
        img_shape = img.shape
        img_conv_out = self.img_conv(img)
        img_vec = img_conv_out.view(img_shape[0], -1)

        theta_conv_out = self.theta_conv(theta)
        theta_vec = theta_conv_out.view(img_shape[0], -1)

        x = torch.cat((img_vec, theta_vec), 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Agent(SynDQNAgent):
    def __init__(self, model, envs, exploration,
                 gamma=0.99, memory_size=100000, batch_size=64, target_update_frequency=1000, saving_dir=None,
                 min_mem=1000):
        saving_dir = '/home/ur5/thesis/rdd_rl/scoop_vision/data/syn_dqn'
        SynDQNAgent.__init__(self, model, envs, exploration, gamma, memory_size, batch_size, target_update_frequency,
                             saving_dir, min_mem)
        if envs is not None:
            self.state_padding = (torch.zeros(self.envs[0].observation_space[0].shape, device=self.device).unsqueeze(0),
                                  torch.zeros(self.envs[0].observation_space[1].shape, device=self.device).unsqueeze(0))

    def getStateFromObs(self, obss):
        states = map(lambda x: (torch.tensor(x[0], device=self.device, dtype=torch.float).unsqueeze(0),
                                torch.tensor(x[1], device=self.device, dtype=torch.float).unsqueeze(0))
                     if x is not None else self.state_padding, obss)
        return states

    def getStateInputTensor(self, states):
        imgs, thetas = zip(*states)
        return torch.cat(imgs), torch.cat(thetas)

    def unzipMemory(self, memory):
        state_0_padding = self.state_padding[0].to('cpu')
        state_1_padding = self.state_padding[1].to('cpu')

        mini_batch = Transition(*zip(*memory))

        state_batch = mini_batch.state
        state_batch = map(lambda x: x if x is not None else (state_0_padding, state_1_padding), state_batch)
        next_state_batch = mini_batch.next_state
        next_state_batch = map(lambda x: x if x is not None else (state_0_padding, state_1_padding), next_state_batch)

        state_0_batch, state_1_batch = zip(*state_batch)
        next_state_0_batch, next_state_1_batch = zip(*next_state_batch)

        state_0 = torch.cat(state_0_batch).to(self.device)
        state_1 = torch.cat(state_1_batch).to(self.device)
        action = torch.cat(mini_batch.action).to(self.device)
        next_state_0 = torch.cat(next_state_0_batch).to(self.device)
        next_state_1 = torch.cat(next_state_1_batch).to(self.device)
        reward = torch.cat(mini_batch.reward).to(self.device)
        final_mask = torch.tensor(mini_batch.final_mask, dtype=torch.uint8).to(self.device)
        pad_mask = torch.tensor(mini_batch.pad_mask, dtype=torch.uint8).to(self.device)
        non_pad_mask = 1 - pad_mask

        return (state_0, state_1), action, (next_state_0, next_state_1), reward, final_mask, non_pad_mask


if __name__ == '__main__':
    envs = []
    for i in range(1):
        env = ScoopEnv(21000 + i)
        envs.append(env)

    agent = Agent(DQN(envs[0].observation_space[0].shape, envs[0].observation_space[1].shape, 4),
                  envs, LinearSchedule(10000, 0.1), batch_size=128, min_mem=200)
    agent.loadCheckpoint('20190221140454', load_memory=False)
    agent.train(100000, 200, 500)

    # agent = Agent(None, None, None)
    # agent.loadCheckpoint('20190221140454', data_only=True)
    # plotLearningCurve(agent.episode_rewards)
    # plt.show()

