import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../..')

from util.utils import LinearSchedule
from agent.drqn_agent_one import OneStepDRQNAgent
from scoop_discrete_sim.scripts.env import SimScoopEnv


class DRQN(torch.nn.Module):
    def __init__(self):
        super(DRQN, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, 5)
        self.conv2 = nn.Conv1d(8, 16, 3)
        self.lstm = nn.LSTMCell(64+4, 16)
        self.fc1 = nn.Linear(16, 4)

        self.hidden_size = 16

    def forward(self, inputs):
        (x, action), (hx, cx) = inputs
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64)
        x = torch.cat((x, action), 1)
        hx, cx = self.lstm(x, (hx, cx))
        x = self.fc1(hx)
        return x, (hx, cx)


class ConvDRQNAgent(OneStepDRQNAgent):
    def __init__(self, model_class, model=None, env=None, exploration=None,
                 gamma=0.99, memory_size=10000, batch_size=128, target_update_frequency=20):
        saving_dir = '/home/ur5/thesis/rdd_rl/scoop_discrete_sim/data/drqn_os_lrud'
        OneStepDRQNAgent.__init__(self, model_class, model, env, exploration, gamma, memory_size, batch_size,
                                  target_update_frequency, saving_dir)
        self.n_action = 4

        self.last_action = [0. for _ in range(self.n_action)]

    def resetEnv(self):
        theta = self.env.reset()
        self.last_action = [0. for _ in range(self.n_action)]
        theta_tensor = torch.tensor(theta, device=self.device).unsqueeze(0).unsqueeze(0)
        action_tensor = torch.tensor(self.last_action, device=self.device).unsqueeze(0)
        self.state = theta_tensor, action_tensor

    def takeAction(self, action):
        action_onehot = np.eye(self.n_action)[action].tolist()
        self.last_action = action_onehot
        return self.env.step(action)

    def getNextState(self, obs):
        theta_tensor = torch.tensor(obs, device=self.device).unsqueeze(0).unsqueeze(0)
        action_tensor = torch.tensor(self.last_action, device=self.device).unsqueeze(0)
        return theta_tensor, action_tensor

    @staticmethod
    def getNonFinalNextStateBatch(mini_batch):
        non_final_next_states = [s for s in mini_batch.next_state
                                 if s is not None]
        theta, action = zip(*non_final_next_states)
        obs = torch.cat(theta)
        action = torch.cat(action)
        return obs, action

    @staticmethod
    def getStateBatch(mini_batch):
        theta, action = zip(*mini_batch.state)
        obs = torch.cat(theta)
        action = torch.cat(action)
        return obs, action

    @staticmethod
    def getNonInitialPreStateBatch(mini_batch):
        non_initial_pre_states = [s for s in mini_batch.pre_state
                                  if s is not None]
        theta, action = zip(*non_initial_pre_states)
        obs = torch.cat(theta)
        action = torch.cat(action)
        return obs, action


if __name__ == '__main__':
    agent = ConvDRQNAgent(DRQN, model=DRQN(), env=SimScoopEnv(),
                          exploration=LinearSchedule(100000, initial_p=1.0, final_p=0.1),
                          batch_size=128)

    agent.train(100000, max_episode_steps=200)