from agent.drqn_agent import DRQNAgent
import torch


class CartPoleDRQNAgent(DRQNAgent):
    def __init__(self, *args, **kwargs):
        DRQNAgent.__init__(self, *args, **kwargs)

    def resetEnv(self):
        """
        reset the env and set self.state
        :return: None
        """
        obs = self.env.reset()
        obs = obs[[0, 2]]
        self.state = torch.tensor(obs, device=self.device, dtype=torch.float).unsqueeze(0)
        return

    def takeAction(self, action):
        """
        take given action and return response
        :param action: int, action to take
        :return: obs_, r, done, info
        """
        obs, r, done, info = self.env.step(action)
        obs = obs[[0, 2]]
        if done:
            r = -1
        return obs, r, done, info

