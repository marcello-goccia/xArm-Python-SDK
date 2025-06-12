import cv2
from primesense import openni2
import numpy as np
import os
import sys
import time

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

# Initialise ROBOT
arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)


def reset_home_position():
    # Set to the original home position angles
    home_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Replace with correct values
    arm.set_servo_angle(angle=home_angles, speed=50, wait=True)

# Call this function whenever you need to reset
reset_home_position()


try:
    # CAMERA
    # Initialize OpenNI2
    openni2.initialize(r"C:\Program Files\Orbbec\OpenNI2\samples\bin")
    dev = openni2.Device.open_any()

    # Start depth stream only
    depth_stream = dev.create_depth_stream()
    depth_stream.start()

    # Start color stream via UVC (OpenCV VideoCapture)
    cap = cv2.VideoCapture(0)

    # Move robot to approximate search position
    arm.set_position(x=300.0, y=200.0, z=400.0, roll=0.0, pitch=90.0, yaw=0.0, speed=50, wait=True)
    time.sleep(2)


    # Capture depth frame
    depth_frame = depth_stream.read_frame()
    depth_data = depth_frame.get_buffer_as_uint16()
    depth_img = np.frombuffer(depth_data, dtype=np.uint16).reshape(480, 640)

    # Capture color frame
    ret, color_img = cap.read()


    # Detect object position (use your color detection logic)
    u = 320
    v = 240
    depth_value_mm = depth_img[v, u]
    print(f"Depth at center: {depth_value_mm} mm")

    # Convert pixel + depth to camera XYZ:
    #  Assuming we have camera intrinsics:
    fx = 525  # example focal length (adjust after calibration)
    fy = 525
    cx = 319.5
    cy = 239.5

    # This gives you 3D position relative to camera.
    Zc = depth_value_mm
    Xc = (u - cx) * Zc / fx
    Yc = (v - cy) * Zc / fy

    # Apply camera offset (since camera is mounted on gripper)
    #  need to know your mounting offset (let’s say for now):
    camera_offset_z = 100  # mm between camera and gripper center
    Xg = Xc
    Yg = Yc
    Zg = Zc - camera_offset_z


    # Convert camera frame to robot frame
    # Since camera is mounted directly on gripper → we may already be in robot frame.
    # (We can refine this part after first tests.)

    # 9️⃣ Move robot to corrected position
    # For test, move forward by the object offset:
    arm.set_position(
        x=300.0 + Xg,
        y=200.0 + Yg,
        z=400.0 - (Zc - 400),  # adjust based on your Z reference
        roll=0.0,
        pitch=90.0,
        yaw=0.0,
        speed=50,
        wait=True
    )

except Exception as e:
    print(f"An exception occurred: {e}")