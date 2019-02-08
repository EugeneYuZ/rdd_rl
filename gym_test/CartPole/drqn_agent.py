from agent.drqn_agent import DRQNAgent


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

