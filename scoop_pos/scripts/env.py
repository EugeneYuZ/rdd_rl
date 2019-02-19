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
        self.sensor = VisionSensor(self.sim_client, 'Vision_sensor', None, None, True, False)
        self.nA = 4

        self.observation_space = np.zeros(4)

        self.cube = None
        self.cube_start_position = [-0.2, 0.85, 0.025]
        self.cube_size = [0.1, 0.2, 0.04]

        self.open_position = 0.3

    def getObs(self):
        sim_ret, theta = utils.getJointPosition(self.sim_client, self.rdd.finger_joint_narrow)
        sim_ret, tip_position = utils.getObjectPosition(self.sim_client, self.ur5.gripper_tip)

        return np.concatenate((tip_position, [theta]))

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

        self.ur5.moveTo(target_pose)

        return self.getObs()

    def step(self, a):
        """
        take a step
        :param a: action, int
        :return: observation, reward, done, info
        """
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
        self.ur5.moveTo(target_pose)

        sim_ret, cube_orientation = utils.getObjectOrientation(self.sim_client, self.cube)
        sim_ret, cube_position = utils.getObjectPosition(self.sim_client, self.cube)
        sim_ret, target_position = utils.getObjectPosition(self.sim_client, self.ur5.UR5_target)

        # arm is in wrong pose
        # sim_ret, target_position = utils.getObjectPosition(self.sim_client, self.ur5.UR5_target)
        if target_position[1] < 0.42 or target_position[1] > 0.95 or target_position[2] < 0 or target_position[
            2] > 0.2:
            # print 'Wrong arm position: ', target_position
            return None, -1, True, None

        # cube in wrong position
        while any(np.isnan(cube_position)):
            res, cube_position = utils.getObjectPosition(self.sim_client, self.cube)
        if cube_position[0] < self.cube_start_position[0] - self.cube_size[0] or \
                cube_position[0] > self.cube_start_position[0] + self.cube_size[0] or \
                cube_position[1] < self.cube_start_position[1] - self.cube_size[1] or \
                cube_position[1] > self.cube_start_position[1] + self.cube_size[1]:
            # print 'Wrong cube position: ', cube_position
            return None, 0, True, None

        # cube is lifted
        if cube_orientation[0] < -0.05:
            return None, 1, True, None

        # cube is not lifted
        return self.getObs(), 0, False, None


if __name__ == '__main__':
    env = ScoopEnv(port=19997)
    env.reset()
    while True:
        a = input('input action')
        s_, r, done, info = env.step(int(a))
        print s_, r, done
        if done:
            break
