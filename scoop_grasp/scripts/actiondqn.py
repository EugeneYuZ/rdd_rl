import sys
sys.path.append('../..')

from util.utils import LinearSchedule
from util.plot import plotLearningCurve
from agent.dqn_agent import *
from scoop_grasp_env import ScoopEnv


class ConvActionDQN(torch.nn.Module):
    def __init__(self):
        super(ConvActionDQN, self).__init__()
        self.conv1 = nn.Conv1d(2, 8, 5)
        self.conv2 = nn.Conv1d(8, 16, 3)
        self.fc1 = nn.Linear(218, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, inputs):
        x, actions = inputs
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = x.view(-1, 208)
        x = torch.cat((x, actions), 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvActionDQNAgent(DQNAgent):
    def __init__(self, model_class, model=None, env=None, exploration=None,
                 gamma=0.99, memory_size=10000, batch_size=128, target_update_frequency=10):
        saving_dir = '/home/ur5/thesis/rdd_rl/scoop_grasp/data/action_dqn'
        DQNAgent.__init__(self, model_class, model, env, exploration, gamma, memory_size, batch_size,
                          target_update_frequency, saving_dir)

        self.action_history = [2 for _ in range(10)]

    def resetEnv(self):
        obs = self.env.reset()
        self.action_history = [2 for _ in range(10)]
        obs_tensor = torch.tensor(obs, device=self.device).unsqueeze(0)
        action_tensor = torch.tensor(self.action_history, device=self.device, dtype=torch.float).unsqueeze(0)
        self.state = obs_tensor, action_tensor

    def takeAction(self, action):
        self.action_history = self.action_history[1:] + [action]
        return self.env.step(action)

    def getNextState(self, obs):
        obs_tensor = torch.tensor(obs, device=self.device).unsqueeze(0)
        action_tensor = torch.tensor(self.action_history, device=self.device, dtype=torch.float).unsqueeze(0)
        return obs_tensor, action_tensor

    @staticmethod
    def getNonFinalNextStateBatch(mini_batch):
        non_final_next_states = [s for s in mini_batch.next_state
                                 if s is not None]
        obs, action = zip(*non_final_next_states)
        obs = torch.cat(obs)
        action = torch.cat(action)
        return obs, action

    @staticmethod
    def getStateBatch(mini_batch):
        obs, action = zip(*mini_batch.state)
        obs = torch.cat(obs)
        action = torch.cat(action)
        return obs, action


if __name__ == '__main__':
    agent = ConvActionDQNAgent(ConvActionDQN, model=ConvActionDQN(), env=ScoopEnv(port=19997),
                               exploration=LinearSchedule(100000, initial_p=1.0, final_p=0.1))
    agent.loadCheckpoint('20190123173739')
    agent.train(100000, max_episode_steps=1000, save_freq=50)