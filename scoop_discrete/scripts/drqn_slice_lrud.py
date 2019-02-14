import sys
import numpy as np
sys.path.append('../..')

from util.utils import LinearSchedule
from util.plot import *
from agent.drqn_slice_agent import *
from scoop_discrete_env_lrud_no_ros import ScoopEnv


class DRQN(torch.nn.Module):
    def __init__(self):
        super(DRQN, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, 5)
        self.conv2 = nn.Conv1d(8, 16, 3)
        self.lstm = nn.LSTM(224+4, 64)
        self.fc1 = nn.Linear(64, 4)

        self.hidden_size = 64

    def forward(self, inputs, hidden=None):
        x, action = inputs
        shape = x.shape
        x = x.view(shape[0]*shape[1], shape[2], shape[3])
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(shape[0], shape[1], -1)
        x = torch.cat((x, action), 2)
        if hidden is None:
            x, hidden = self.lstm(x)
        else:
            x, hidden = self.lstm(x, hidden)
        x = self.fc1(x)
        return x, hidden


class ConvDRQNAgent(DRQNSliceAgent):
    def __init__(self, model_class, model=None, env=None, exploration=None, batch_size=32, sequence_len=10,
                 **kwargs):
        saving_dir = '/home/ur5/thesis/rdd_rl/scoop_discrete/data/drqn_slice_lrud'
        DRQNSliceAgent.__init__(self, model_class, model, env, exploration,
                                batch_size=batch_size, sequence_len=sequence_len, saving_dir=saving_dir, **kwargs)
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

    def forwardPolicyNet(self, state):
        with torch.no_grad():
            obs = state[0]
            act = state[1]
            obs = obs.unsqueeze(0)
            act = act.unsqueeze(0)
            q_values, self.hidden = self.policy_net((obs, act), self.hidden)
            q_values = q_values.squeeze(0)
            return q_values

    def unzipMemory(self, memory):
        state_1_batch = []
        state_2_batch = []

        next_state_1_batch = []
        next_state_2_batch = []

        action_batch = []
        reward_batch = []
        final_mask_batch = []
        pad_mask_batch = []

        state_1_padding = self.state_padding
        state_2_padding = torch.tensor([0. for _ in range(self.n_action)]).unsqueeze(0)

        for episode in memory:
            episode_transition = Transition(*zip(*episode))
            state_1_batch.append(torch.cat([s[0] if s is not None else state_1_padding
                                            for s in episode_transition.state]))
            state_2_batch.append(torch.cat([s[1] if s is not None else state_2_padding
                                            for s in episode_transition.state]))
            next_state_1_batch.append(torch.cat([s[0] if s is not None else state_1_padding
                                                 for s in episode_transition.next_state]))
            next_state_2_batch.append(torch.cat([s[1] if s is not None else state_2_padding
                                                 for s in episode_transition.next_state]))

            action_batch.append(torch.cat(episode_transition.action))
            reward_batch.append(torch.cat(episode_transition.reward))
            final_mask_batch.append(torch.tensor(list(episode_transition.final_mask), dtype=torch.uint8))
            pad_mask_batch.append(torch.tensor(list(episode_transition.pad_mask), dtype=torch.uint8))

        state = (torch.stack(state_1_batch).to(self.device),
                 torch.stack(state_2_batch).to(self.device))
        action = torch.stack(action_batch).to(self.device)
        next_state = (torch.stack(next_state_1_batch).to(self.device),
                      torch.stack(next_state_2_batch).to(self.device))
        reward = torch.stack(reward_batch).to(self.device)
        final_mask = torch.stack(final_mask_batch).to(self.device)
        pad_mask = torch.stack(pad_mask_batch)
        non_pad_mask = 1 - pad_mask

        return state, action, next_state, reward, final_mask, non_pad_mask


if __name__ == '__main__':
    agent = ConvDRQNAgent(DRQN, model=DRQN(), env=ScoopEnv(),
                          exploration=LinearSchedule(10000, initial_p=1.0, final_p=0.1), min_mem=1000)
    agent.loadCheckpoint('20190212192630')
    agent.train(100000, max_episode_steps=200)
