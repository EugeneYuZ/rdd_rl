from agent.drqn_agent import DRQNAgent
import numpy as np

class CartPoleDRQNAgent(DRQNAgent):
    def __init__(self, *args, **kwargs):
        DRQNAgent.__init__(self, *args, **kwargs)

    def takeAction(self, action):
        """
        take given action and return response
        :param action: int, action to take
        :return: obs_, r, done, info
        """
        obs, r, done, info = self.env.step(action)
        if done:
            r = -1
        return obs, r, done, info

    def train(self, num_episodes, max_episode_steps=100, save_freq=100, render=False, print_step=True):
        while self.episodes_done < num_episodes:
            self.trainOneEpisode(num_episodes, max_episode_steps, save_freq, render, print_step)
            if len(self.episode_rewards) > 100:
                avg = np.average(self.episode_rewards[-100:])
                print 'avg reward in 100 episodes: ', avg
                if avg > 195:
                    print 'solved'
                    return
