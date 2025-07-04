import sys, os
import time
import math
import numpy as np
from xarm.wrapper import XArmAPI
from configparser import ConfigParser

path_ip_robot = '../../robot.conf'
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))


class Robot:
    ip = None
    home_joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def __init__(self):
        self.read_ip_robot()
        # initialise the robot
        self.arm = XArmAPI(self.ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)

        self.wait()
        self.reset_home_position()

    def read_ip_robot(self):
        try:
            parser = ConfigParser()
            parser.read(path_ip_robot)
            self.ip = parser.get('xArm', 'ip')
        except:
            self.ip = input('Please input the xArm ip address:')
            if not self.ip:
                print('input error, exit')
                sys.exit(1)

    def disconnect_robot(self):
        try:
            self.arm.disconnect()
        except Exception:
            pass

    def reset_home_position(self):
        # Set to the original home position angles
        self.arm.set_servo_angle(
            angle= self.home_joint_angles,
            speed= 50,
            wait= True)

    def wait(self, secs=1):
        time.sleep(secs)

    def set_initial_tracking_position(self):
        print("Move robot to starting point")

        robot_x, robot_y, robot_z = 266.0, 0.0, 303.0

        roll, pitch, yaw = -126, -88, -55

        self.arm.set_position(x=robot_x, y=robot_y, z=robot_z,
                              roll=roll, pitch=pitch, yaw=yaw,
                              speed=100, wait=True)
        self.wait()

    def get_robot_position(self, is_radian=False):
        return self.arm.get_position(is_radian)

    def get_robot_servo_angles(self):
        return self.arm.get_servo_angle()

    def set_robot_position(self, x, y, z, roll=False, pitch=False, yaw=False, speed=50, wait=True):
        if not roll and not pitch and not yaw:
            self.arm.set_position(x=x, y=y, z=z,
                                  speed=speed, wait=wait)
        else:
            self.arm.set_position(x=x, y=y, z=z,
                                  roll=roll, pitch=pitch, yaw=yaw,
                                  speed=speed, wait=wait)

    def pose_to_transform(self, x, y, z, roll, pitch, yaw):
        """
        Build a 4×4 matrix from translation (mm) + RPY in degrees.
        Returns T (base→gripper) or similarly for any frame.
        """
        # Convert degrees→radians
        r, p, w = map(math.radians, (roll, pitch, yaw))
        # Rotation around X, Y, Z
        Rx = np.array([[1,       0,       0],
                       [0, math.cos(r), -math.sin(r)],
                       [0, math.sin(r),  math.cos(r)]])
        Ry = np.array([[ math.cos(p), 0, math.sin(p)],
                       [          0, 1,          0],
                       [-math.sin(p), 0, math.cos(p)]])
        Rz = np.array([[math.cos(w), -math.sin(w), 0],
                       [math.sin(w),  math.cos(w), 0],
                       [         0,           0, 1]])
        R = Rz.dot(Ry).dot(Rx)
        T = np.eye(4, dtype=float)
        T[0:3, 0:3] = R
        T[0:3, 3]   = [x, y, z]

        return T
