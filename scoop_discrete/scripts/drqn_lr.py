import sys
import numpy as np
sys.path.append('../..')

from util.utils import LinearSchedule
from util.plot import plotLearningCurve
from agent.drqn_agent import *
from scoop_discrete_env_lr_no_ros import ScoopEnv


class DRQN(torch.nn.Module):
    def __init__(self):
        super(DRQN, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, 5)
        self.conv2 = nn.Conv1d(8, 16, 3)
        self.lstm = nn.LSTMCell(224+2, 64)
        self.fc1 = nn.Linear(64, 2)

        self.hidden_size = 64

    def forward(self, inputs):
        (x, action), (hx, cx) = inputs
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 224)
        x = torch.cat((x, action), 1)
        hx, cx = self.lstm(x, (hx, cx))
        x = F.relu(hx)
        x = self.fc1(x)
        return x, (hx, cx)


class ConvDRQNAgent(DRQNAgent):
    def __init__(self, model_class, model=None, env=None, exploration=None,
                 gamma=0.99, memory_size=1000, batch_size=1, target_update_frequency=20):
        saving_dir = '/home/ur5/thesis/rdd_rl/scoop_discrete/data/drqn_lr'
        DRQNAgent.__init__(self, model_class, model, env, exploration, gamma, memory_size, batch_size,
                           target_update_frequency, saving_dir)
        self.n_action = 2

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


if __name__ == '__main__':
    agent = ConvDRQNAgent(DRQN, model=DRQN(), env=ScoopEnv(),
                          exploration=LinearSchedule(100000, initial_p=1.0, final_p=0.1),
                          batch_size=4)
    agent.loadCheckpoint('20190205214423')
    agent.train(100000, max_episode_steps=200)
