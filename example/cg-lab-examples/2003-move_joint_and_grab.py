#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2019, UFACTORY, Inc.
# All rights reserved.
#
# Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

"""
Description: Move Joint
"""

import os
import sys
import time
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI


#######################################################
"""
Just for test example
"""
if len(sys.argv) >= 2:
    ip = sys.argv[1]
else:
    try:
        from configparser import ConfigParser
        parser = ConfigParser()
        parser.read('../robot.conf')
        ip = parser.get('xArm', 'ip')
    except:
        ip = input('Please input the xArm ip address:')
        if not ip:
            print('input error, exit')
            sys.exit(1)
########################################################


arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

number_of_loops = 10
move_arm = True

def reset_home_position():
    # Set to the original home position angles
    home_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Replace with correct values
    arm.set_servo_angle(angle=home_angles, speed=50, wait=True)

# Call this function whenever you need to reset
reset_home_position()

for i in range(number_of_loops):

    ## arm.set_gripper_position(400, speed=1000, wait=True)
    print(f"loop run {i} times")

    arm.move_gohome(wait=True)

    if move_arm:

        # # arm.set_position(x=300.0, y=200.0, z=400.0, roll=0.0, pitch=90.0, yaw=0.0, speed=50, wait=True)
        arm.set_position(x=300.0, y=200.0, z=400.0, roll=0.0, pitch=90.0, yaw=0.0, speed=300, wait=True)

        time.sleep(5)

        arm.set_position(x=400.0, y=200.0, z=400.0, roll=0.0, pitch=90.0, yaw=0.0, speed=300, wait=True)

    ## arm.set_gripper_position(200, speed=1000, wait=True)

    time.sleep(5)

print("process finished")
arm.disconnect()
