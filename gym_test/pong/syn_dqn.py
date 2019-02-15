import sys
sys.path.append('../..')
from agent.syn_agent.syn_dqn_agent import *
from util.plot import *


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)

        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))

        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.float() / 256
        conv_out = self.conv(x)
        x = conv_out.view(x.size()[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class DQNStackAgent(SynDQNAgent):
    def __init__(self, *args, **kwargs):
        SynDQNAgent.__init__(self, *args, **kwargs)

    def resetEnv(self):
        def reset(env):
            return env.reset()
        obss = self.pool.map(reset, self.envs)
        obss = map(lambda x: np.array(x), obss)
        self.alive_idx = [i for i in range(self.n_env)]
        states = map(lambda x: torch.tensor(x, device=self.device, dtype=torch.float).unsqueeze(0), obss)
        return states

    def getNextState(self, obss):
        obss = map(lambda x: np.array(x), obss)
        next_states = map(lambda x: torch.tensor(x, device=self.device, dtype=torch.float).unsqueeze(0)
                          if x is not None else None, obss)
        return next_states


if __name__ == '__main__':
    envs = []
    for i in range(4):
        env = gym.make('PongNoFrameskip-v4')
        env = wrap_dqn(env)
        env.seed(i)
        envs.append(env)

    agent = DQNStackAgent(DQN(envs[0].observation_space.shape, envs[0].action_space.n), envs, exploration=LinearSchedule(100000, 0.02),
                          batch_size=128, target_update_frequency=1000, memory_size=100000, min_mem=10000)
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/gym_test/pong/data/syn_dqn'
    agent.train(10000, 10000)
