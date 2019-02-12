import numpy as np
import unittest


class SimScoopEnv:
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3

    def __init__(self):
        self.nA = 4
        self.observation_space = np.zeros((1, 10))

        self.x_lim = [0, 10]
        self.y_lim = [0, 10]

        self.block_size = [4, 1]
        self.obstacle_size = [2, 1]

        self.x = 0.
        self.y = 0.

    def reset(self):
        self.x = float(np.random.randint(2, 9))
        self.y = float(np.random.randint(2, 9))

        return [0. for _ in range(10)]

    def step(self, a):
        if a == self.RIGHT:
            if self.x >= 9:
                return None, -1, True, None
            obs = [self.y for _ in range(10)]
            if self.x < 2:
                obs = [self.y - 1 for _ in range(10)]
            elif self.x == 2:
                obs = [self.y - 1 for _ in range(5)]
                obs += np.linspace(self.y - 1, self.y, 5).tolist()
            elif self.x == 3:
                obs = np.linspace(self.y - 1, self.y, 5).tolist()
                obs += [self.y for _ in range(5)]
            elif 3 < self.x < 7:
                obs = [self.y for _ in range(10)]
            elif self.x == 7:
                obs = [self.y for _ in range(5)]
                obs += np.linspace(self.y, self.y - 1, 5).tolist()
            elif self.x == 8:
                obs = np.linspace(self.y, self.y - 1, 5).tolist()
                obs += [self.y - 1 for _ in range(5)]
            self.x += 2
            return obs, 0, False, None

        elif a == self.LEFT:
            if self.x <= 5 and self.y == 1:
                return None, +1, True, None
            elif self.x <= 1:
                return None, -1, True, None
            obs = [self.y for _ in range(10)]
            if self.x == 10:
                obs = [self.y - 1 for _ in range(5)]
                obs += np.linspace(self.y - 1, self.y, 5).tolist()
            elif self.x == 9:
                obs = np.linspace(self.y - 1, self.y, 5).tolist()
                obs += [self.y for _ in range(5)]
            elif 5 < self.x < 9:
                obs = [self.y for _ in range(10)]
            elif self.x == 5:
                obs = [self.y for _ in range(5)]
                obs += np.linspace(self.y, self.y - 1, 5).tolist()
            elif self.x == 4:
                obs = np.linspace(self.y, self.y - 1, 5).tolist()
                obs += [self.y - 1 for _ in range(5)]
            elif self.x < 4:
                obs = [self.y - 1 for _ in range(10)]
            self.x -= 2
            return obs, 0, False, None

        elif a == self.UP:
            if self.y == 10:
                return None, -1, True, None
            else:
                if 4 <= self.x <= 8:
                    obs = np.linspace(self.y, self.y + 1, 10).tolist()
                else:
                    obs = np.linspace(self.y - 1, self.y, 10).tolist()
                self.y += 1
                return obs, 0, False, None

        elif a == self.DOWN:
            if self.y == 1:
                if 4 <= self.x <= 8:
                    obs = [self.y for _ in range(10)]
                else:
                    obs = [self.y - 1 for _ in range(10)]
                return obs, 0, False, None

            else:
                if 4 <= self.x <= 8:
                    obs = np.linspace(self.y, self.y - 1, 10).tolist()
                else:
                    obs = np.linspace(self.y - 1, self.y - 2, 10).tolist()

                self.y -= 1
                return obs, 0, False, None


env = SimScoopEnv()


class Test(unittest.TestCase):

    def testUp(self):
        env.x = 4
        env.y = 9
        obs, r, done, info = env.step(env.UP)
        self.assertEqual(env.y, 10)
        self.assertEqual(obs, np.linspace(9, 10, 10).tolist())

        env.x = 3
        env.y = 9
        obs, r, done, info = env.step(env.UP)
        self.assertEqual(env.y, 10)
        self.assertEqual(obs, np.linspace(8, 9, 10).tolist())

    def testDown(self):
        env.x = 4
        env.y = 9
        obs, r, done, info = env.step(env.DOWN)
        self.assertEqual(env.y, 8)
        self.assertEqual(obs, np.linspace(9, 8, 10).tolist())

        env.x = 3
        env.y = 9
        obs, r, done, info = env.step(env.DOWN)
        self.assertEqual(env.y, 8)
        self.assertEqual(obs, np.linspace(8, 7, 10).tolist())

        env.y = 1
        obs, r, done, info = env.step(env.DOWN)
        self.assertEqual(env.y, 1)
        self.assertEqual(obs, [0 for _ in range(10)])

        env.x = 8
        obs, r, done, info = env.step(env.DOWN)
        self.assertEqual(env.y, 1)
        self.assertEqual(obs, [1 for _ in range(10)])

    def testLeft(self):
        env.x = 5
        env.y = 1
        obs, r, done, info = env.step(env.LEFT)
        self.assertEqual(r, 1)

        env.x = 5
        env.y = 2
        obs, r, done, info = env.step(env.LEFT)
        self.assertEqual(obs, [2 for _ in range(5)] + np.linspace(2, 1, 5).tolist())
        self.assertEqual(env.x, 3)

        obs, r, done, info = env.step(env.LEFT)
        self.assertEqual(obs, [1 for _ in range(10)])
        self.assertEqual(env.x, 1)

        obs, r, done, info = env.step(env.LEFT)
        self.assertTrue(done)

        env.x = 4
        obs, r, done, info = env.step(env.LEFT)
        self.assertEqual(obs, np.linspace(2, 1, 5).tolist() + [1 for _ in range(5)])
        self.assertEqual(env.x, 2)










if __name__ == '__main__':
    # env = SimScoopEnv()
    # env.reset()
    # while True:
    #     print env.x, env.y
    #     a = input('input action')
    #     s_, r, done, info = env.step(int(a))
    #     print s_, r, done
    #     if done:
    #         break

    unittest.main()



