import sys
sys.path.append('../..')

from util.utils import LinearSchedule
from util.plot import plotLearningCurve
from agent.dqn_agent import *
from scoop_grasp_env import ScoopEnv


class ConvDQN(torch.nn.Module):
    def __init__(self):
        super(ConvDQN, self).__init__()
        self.conv1 = nn.Conv1d(2, 8, 5)
        self.conv2 = nn.Conv1d(8, 16, 3)
        self.fc1 = nn.Linear(864, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 864)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvDQNAgent(DQNAgent):
    def __init__(self, model_class, model=None, env=None, exploration=None,
                 gamma=0.99, memory_size=10000, batch_size=64, target_update_frequency=10):
        saving_dir = '/home/ur5/thesis/rdd_rl/scoop_grasp/data/dqn'
        DQNAgent.__init__(self, model_class, model, env, exploration, gamma, memory_size, batch_size,
                          target_update_frequency, saving_dir)


if __name__ == '__main__':
    # agent = Agent(DQN, model=DQN(), env=ScoopEnv(port=19997),
    #               exploration=LinearSchedule(100000, initial_p=1.0, final_p=0.1), batch_size=128)
    # agent.train(100000, max_episode_steps=200)

    agent = ConvDQNAgent(ConvDQN)
    agent.load_checkpoint('20190123164517')
    plotLearningCurve(agent.episode_rewards)
    plt.show()
    plotLearningCurve(agent.episode_lengths, label='length', color='r')
    plt.show()
