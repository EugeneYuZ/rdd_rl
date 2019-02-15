import sys
sys.path.append('../..')
from agent.syn_agent.syn_drqn_slice_agent import *


class DRQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DRQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)

        self.lstm = nn.LSTM(conv_out_size, 512, batch_first=True)
        self.fc = nn.Linear(512, n_actions)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))

        return int(np.prod(o.size()))

    def forward(self, x, hidden=None):
        x = x.float() / 256
        shape = x.shape
        x = x.view(shape[0]*shape[1], shape[2], shape[3], shape[4])
        conv_out = self.conv(x)
        x = conv_out.view(shape[0], shape[1], -1)
        if hidden is None:
            x, hidden = self.lstm(x)
        else:
            x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden


class DRQNStackAgent(SynDRQNAgent):
    def __init__(self, *args, **kwargs):
        SynDRQNAgent.__init__(self, *args, **kwargs)

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
        env = wrap_drqn(env)
        env.seed(i)
        envs.append(env)

    agent = DRQNStackAgent(DRQN(envs[0].observation_space.shape, envs[0].action_space.n), envs, exploration=LinearSchedule(100000, 0.02),
                          batch_size=128, target_update_frequency=1000, memory_size=100000, min_mem=10000, sequence_len=10)
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/gym_test/pong/data/syn_drqn'
    agent.train(10000, 10000, 200)