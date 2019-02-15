import os
import torch


class PlotAgent:
    def __init__(self):
        self.saving_dir = None
        self.episode_rewards = None
        self.episode_lengths = None

    def loadCheckpoint(self, time_stamp):
        state_filename = os.path.join(self.saving_dir, 'checkpoint.' + time_stamp + '.pth.tar')

        print 'loading checkpoint: ', time_stamp
        checkpoint = torch.load(state_filename)
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        return
