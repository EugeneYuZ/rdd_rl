import time
import numpy as np
import gym.spaces as spaces

from vrep_arm_toolkit.simulation import vrep
from vrep_arm_toolkit.robots.ur5 import UR5
from vrep_arm_toolkit.grippers.rdd import RDD
from vrep_arm_toolkit.sensors.vision_sensor import VisionSensor
import vrep_arm_toolkit.utils.vrep_utils as utils
from vrep_arm_toolkit.utils import transformations


class ScoopEnv:
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3

    def __init__(self, port=19997):
        np.random.seed(port)

        self.sim_client = utils.connectToSimulation('127.0.0.1', port)

        # Create UR5 and restart simulator
        self.rdd = RDD(self.sim_client)
        self.ur5 = UR5(self.sim_client, self.rdd)
        self.sensor = VisionSensor(self.sim_client, 'Vision_sensor_top', None, None, True, False)
        self.nA = 4

        self.observation_space = (np.zeros((12, 64, 64)), np.zeros((4, 20)))

        self.cube = None
        self.cube_start_position = [-0.2, 0.85, 0.025]
        self.cube_size = [0.1, 0.2, 0.04]

        self.open_position = 0.3

        self.img_his = [np.zeros((3, 64, 64)) for _ in range(4)]
        self.theta_his = [np.zeros((1, 20)) for _ in range(4)]

    def sendClearSignal(self):
        sim_ret = vrep.simxSetIntegerSignal(self.sim_client, 'clear', 1, utils.VREP_ONESHOT)

    def getObs(self):
        sim_ret, data = vrep.simxGetStringSignal(self.sim_client, 'theta', vrep.simx_opmode_blocking)
        p = vrep.simxUnpackFloats(data)
        if len(p) == 0:
            p = [0.]
        xs = [i for i in range(len(p))]
        resampled = np.interp(np.linspace(0, len(p) - 1, 20), xs, p)

        img_obs = np.rollaxis(self.sensor.getColorData(), 2, 0)
        theta_obs = np.expand_dims(resampled, 0)

        self.img_his = self.img_his[1:] + [img_obs]
        self.theta_his = self.theta_his[1:] + [theta_obs]

        return np.concatenate(self.img_his, 0), np.concatenate(self.theta_his, 0)

    def reset(self):
        """
        reset the environment
        :return: the observation, List[List[float], List[float]]
        """
        vrep.simxStopSimulation(self.sim_client, utils.VREP_BLOCKING)
        time.sleep(1)
        vrep.simxStartSimulation(self.sim_client, utils.VREP_BLOCKING)
        time.sleep(1)

        sim_ret, self.cube = utils.getObjectHandle(self.sim_client, 'cube')

        # utils.setObjectPosition(self.sim_client, self.ur5.UR5_target, [-0.2, 0.6, 0.08])
        utils.setObjectPosition(self.sim_client, self.ur5.UR5_target, [-0.2, 0.6, 0.15])

        dy = 0.3 * np.random.random()
        dz = 0.1 * np.random.random() - 0.05
        current_pose = self.ur5.getEndEffectorPose()
        target_pose = current_pose.copy()
        target_pose[1, 3] += dy
        target_pose[2, 3] += dz

        self.rdd.setFingerPos()

        self.sendClearSignal()
        self.ur5.moveTo(target_pose)

        return self.getObs()

    def getReward(self):
        sim_ret, narrow_tip = utils.getObjectHandle(self.sim_client, 'narrow_tip')
        sim_ret, cube_bottom = utils.getObjectHandle(self.sim_client, 'cube_bottom')

        sim_ret, tip_position = utils.getObjectPosition(self.sim_client, narrow_tip)
        sim_ret, bottom_position = utils.getObjectPosition(self.sim_client, cube_bottom)
        sim_ret, cube_orientation = utils.getObjectOrientation(self.sim_client, self.cube)

        return -np.linalg.norm(tip_position-bottom_position) + (-10 * cube_orientation[0])

    def step(self, a):
        """
        take a step
        :param a: action, int
        :return: observation, reward, done, info
        """
        self.sendClearSignal()
        sim_ret, target_position = utils.getObjectPosition(self.sim_client, self.ur5.UR5_target)
        sim_ret, target_orientation = utils.getObjectOrientation(self.sim_client, self.ur5.UR5_target)
        target_pose = transformations.euler_matrix(target_orientation[0], target_orientation[1], target_orientation[2])
        target_pose[:3, -1] = target_position

        if a == self.RIGHT:
            target_pose[1, 3] -= 0.05
        elif a == self.LEFT:
            target_pose[1, 3] += 0.05
        elif a == self.UP:
            target_pose[2, 3] += 0.03
        elif a == self.DOWN:
            target_pose[2, 3] -= 0.03

        target_position = target_pose[:, 3]
        if 0.42 < target_position[1] < 0.95 and 0 < target_position[2] < 0.3:
            self.ur5.moveTo(target_pose)

        sim_ret, cube_orientation = utils.getObjectOrientation(self.sim_client, self.cube)
        sim_ret, cube_position = utils.getObjectPosition(self.sim_client, self.cube)

        # cube in wrong position
        while any(np.isnan(cube_position)):
            res, cube_position = utils.getObjectPosition(self.sim_client, self.cube)
        if not (self.cube_start_position[0] - self.cube_size[0] < cube_position[0]
                < self.cube_start_position[0] + self.cube_size[0] and
                self.cube_start_position[1] - self.cube_size[1] < cube_position[1]
                < self.cube_start_position[1] + self.cube_size[1] and
                self.cube_start_position[2] - self.cube_size[2] < cube_position[2]
                < self.cube_start_position[2] + self.cube_size[2] and
                cube_orientation[0] < 0.1):
            # print 'Wrong cube position: ', cube_position
            return None, -10, True, None

        # cube is lifted
        if cube_orientation[0] < -0.02:
            return None, self.getReward(), True, None

        # cube is not lifted
        return self.getObs(), self.getReward(), False, None


if __name__ == '__main__':
    env = ScoopEnv(port=21000)
    env.reset()
    while True:
        a = input('input action')
        s_, r, done, info = env.step(int(a))
        print s_, r, done
        if done:
            break
