import sys
import numpy as np
sys.path.append('../..')

from util.utils import LinearSchedule
from util.plot import plotLearningCurve
from agent.dqn_agent import *
from scoop_grasp_env_discrete import ScoopEnv


class ConvDQN(torch.nn.Module):
    def __init__(self):
        super(ConvDQN, self).__init__()
        self.conv1 = nn.Conv1d(3, 8, 5)
        self.conv2 = nn.Conv1d(8, 16, 3)
        self.fc1 = nn.Linear(242, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, inputs):
        x, actions, targets = inputs
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 224)
        x = torch.cat((x, actions, targets), 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvDQNAgent(DQNAgent):
    def __init__(self, model_class, model=None, env=None, exploration=None,
                 gamma=0.99, memory_size=10000, batch_size=128, target_update_frequency=10):
        saving_dir = '/home/ur5/thesis/rdd_rl/scoop_discrete/data/action_dqn'
        DQNAgent.__init__(self, model_class, model, env, exploration, gamma, memory_size, batch_size,
                          target_update_frequency, saving_dir)

        self.n_action = 4
        self.n_history = 3

        self.theta_history = [[0. for _ in range(20)] for _ in range(self.n_history)]
        self.action_history = [0. for _ in range(self.n_history * self.n_action)]
        self.theta_target_history = [0. for _ in range(2 * self.n_history)]

        self.current_target = 0

    def resetEnv(self):
        theta = self.env.reset()
        self.theta_history = self.theta_history[1:] + [theta]
        self.action_history = [0. for _ in range(self.n_history * self.n_action)]
        self.theta_target_history = [0. for _ in range(2 * self.n_history)]
        theta_tensor = torch.tensor(self.theta_history, device=self.device).unsqueeze(0)
        action_tensor = torch.tensor(self.action_history, device=self.device).unsqueeze(0)
        target_tensor = torch.tensor(self.theta_target_history, device=self.device).unsqueeze(0)

        self.state = theta_tensor, action_tensor, target_tensor

    def takeAction(self, action):
        action_onehot = np.eye(self.n_action)[action].tolist()
        self.action_history = self.action_history[self.n_action:] + action_onehot
        if action == self.env.OPEN:
            target = 1
        elif action == self.env.CLOSE:
            target = 0
        else:
            target = self.current_target
        self.current_target = target
        target_onehot = np.eye(2)[target].tolist()
        self.theta_target_history = self.theta_target_history[2:] + target_onehot
        return self.env.step(action)

    def getNextState(self, obs):
        self.theta_history = self.theta_history[1:] + [obs]
        theta_tensor = torch.tensor(self.theta_history, device=self.device).unsqueeze(0)
        action_tensor = torch.tensor(self.action_history, device=self.device).unsqueeze(0)
        target_tensor = torch.tensor(self.theta_target_history, device=self.device).unsqueeze(0)
        return theta_tensor, action_tensor, target_tensor

    @staticmethod
    def getNonFinalNextStateBatch(mini_batch):
        non_final_next_states = [s for s in mini_batch.next_state
                                 if s is not None]
        theta, action, target = zip(*non_final_next_states)
        obs = torch.cat(theta)
        action = torch.cat(action)
        target = torch.cat(target)
        return obs, action, target

    @staticmethod
    def getStateBatch(mini_batch):
        theta, action, target = zip(*mini_batch.state)
        obs = torch.cat(theta)
        action = torch.cat(action)
        target = torch.cat(target)
        return obs, action, target


if __name__ == '__main__':
    agent = ConvDQNAgent(ConvDQN, model=ConvDQN(), env=ScoopEnv(port=19997),
                         exploration=LinearSchedule(100000, initial_p=1.0, final_p=0.1))
    agent.train(100000, max_episode_steps=200)
