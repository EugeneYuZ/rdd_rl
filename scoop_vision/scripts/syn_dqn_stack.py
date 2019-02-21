import sys
import numpy as np

from syn_dqn import DQN, Agent

sys.path.append('../..')

from util.utils import LinearSchedule
from util.plot import *
from agent.syn_agent.syn_dqn_agent import *
from env_dense_r_stack import ScoopEnv

if __name__ == '__main__':
    envs = []
    for i in range(8):
        env = ScoopEnv(19997 + i)
        envs.append(env)

    agent = Agent(DQN(envs[0].observation_space[0].shape, envs[0].observation_space[1].shape, 4),
                  envs, LinearSchedule(10000, 0.1), batch_size=128, min_mem=200)
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/scoop_vision/data/syn_dqn_stack'
    agent.train(100000, 200, 500)

    # agent = Agent(None, None, None)
    # agent.loadCheckpoint('20190221140454', data_only=True)
    # plotLearningCurve(agent.episode_rewards)
    # plt.show()

