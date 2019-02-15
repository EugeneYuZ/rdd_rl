import sys
sys.path.append('../..')
from agent.plot_agent import PlotAgent
from util.plot import *

if __name__ == '__main__':
    syn_drqn = PlotAgent()
    syn_drqn.saving_dir = '/home/ur5/thesis/rdd_rl/gym_test/pong/data/syn_drqn'
    syn_drqn.loadCheckpoint('20190215141952')

    drqn = PlotAgent()
    drqn.saving_dir = '/home/ur5/thesis/rdd_rl/gym_test/pong/data/drqn_slice'
    drqn.loadCheckpoint('20190212140951')

    plotLearningCurve(syn_drqn.episode_rewards)
    plotLearningCurve(drqn.episode_rewards, color='r')
    plt.show()
