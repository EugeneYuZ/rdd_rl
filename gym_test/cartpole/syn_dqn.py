import sys
sys.path.append('../..')
from agent.syn_agent.syn_dqn_agent import *


class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Agent(SynDQNAgent):
    def __init__(self, *args, **kwargs):
        SynDQNAgent.__init__(self, *args, **kwargs)

    @staticmethod
    def _act(args):
        (env, action) = args
        obs, r, done, info = env.step(action)
        if done:
            r = -1
        return obs, r, done, info

    def train(self, num_episodes, max_episode_steps=100, save_freq=100):
        while self.episodes_done < num_episodes:
            self.trainOneEpisode(num_episodes, max_episode_steps, save_freq)
            if len(self.episode_rewards) > 100:
                avg = np.average(self.episode_rewards[-100:])
                print 'avg reward in 100 episodes: ', avg
                if avg > 195:
                    print 'solved'
                    return


if __name__ == '__main__':
    envs = []
    for i in range(4):
        env = gym.make("CartPole-v1")
        env.seed(i)
        envs.append(env)

    agent = Agent(DQN(), envs, LinearSchedule(10000, 0.02), batch_size=128, min_mem=1000)
    agent.train(10000, 500)