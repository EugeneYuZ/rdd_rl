import sys
sys.path.append('../..')

from agent.plot_agent import *
from util.plot import *

if __name__ == '__main__':
    agent = PlotAgent()
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/scoop_vision/data/cmp'
    agent.loadCheckpoint('pos')
    plotLearningCurve(agent.episode_rewards[:3000], label='pos')
    agent.loadCheckpoint('top')
    plotLearningCurve(agent.episode_rewards[:3000], label='top', color='r')
    agent.loadCheckpoint('wrist')
    agent.episode_rewards[671] = 0
    plotLearningCurve(agent.episode_rewards[:3000], label='wrist', color='g')
    agent.loadCheckpoint('stack')
    plotLearningCurve(agent.episode_rewards[:3000], label='stack', color='yellow')

    plt.ylim(bottom=-20)
    plt.show()
