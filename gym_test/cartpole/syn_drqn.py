import sys
sys.path.append('../..')
from agent.syn_agent.syn_drqn_slice_agent import *


class DRQN(torch.nn.Module):
    def __init__(self):
        super(DRQN, self).__init__()

        self.fc1 = nn.Linear(4, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x, hidden=None):
        x = F.relu(self.fc1(x))
        if hidden is None:
            x, hidden = self.lstm(x)
        else:
            x, hidden = self.lstm(x, hidden)
        x = self.fc2(x)
        return x, hidden


class CartPoleDRQNAgent(SynDRQNAgent):
    def __init__(self, *args, **kwargs):
        SynDRQNAgent.__init__(self, *args, **kwargs)

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

    agent = CartPoleDRQNAgent(DRQN(), envs, LinearSchedule(10000, 0.02), batch_size=128, min_mem=1000, sequence_len=32)
    agent.train(10000, 500)