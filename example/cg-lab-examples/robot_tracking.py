import cv2
from primesense import openni2
import numpy as np
import os
import sys
import time
from log_code import get_debug_info

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI

#################### ROBOT INIT ####################
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

print("ROBOT INIT - DONE")

#################### CAMERA INIT ####################
try:
    print("STARTING CAMERA INIT")

    openni2.initialize(r"C:\Program Files\Orbbec\OpenNI2\samples\bin")
    dev = openni2.Device.open_any()
    print("Capturing depth camera")
    depth_stream = dev.create_depth_stream()
    depth_stream.start()

    print("Capturing rgb camera")
    cap = cv2.VideoCapture(0)

    print("Move robot to starting point")
    robot_x, robot_y, robot_z = 300.0, 200.0, 400.0
    arm.set_position(x=robot_x, y=robot_y, z=robot_z, roll=0.0, pitch=90.0, yaw=0.0, speed=50, wait=True)
    time.sleep(2)

    print("CAMERA INIT - DONE")

except Exception as e:
    print(f"An exception occurred during camera initialisation: {e} - {get_debug_info}")

def read_region_depth(u, v):
    window_size = 5  # 5x5 window
    half_window = window_size // 2

    # Extract window around (u,v)
    depth_window = depth_img[
                   max(0, v - half_window):min(480, v + half_window),
                   max(0, u - half_window):min(640, u + half_window)
                   ]

    # Filter out zero depths
    valid_depths = depth_window[depth_window > 0]

    if valid_depths.size == 0:
        return 0

    # Use median depth (more stable)
    return np.median(valid_depths)



#################### INITIAL OBJECT DETECTION ####################
def initial_object_detection():
    # Grab frames
    print("Starting initial object detection")

    depth_frame = depth_stream.read_frame()
    depth_data = depth_frame.get_buffer_as_uint16()
    depth_img = np.frombuffer(depth_data, dtype=np.uint16).reshape(480, 640)
    ret, color_img = cap.read()

    # Detect red object (basic color detection)
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = mask1 + mask2

    # Call findContours first!
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        bbox = (x, y, w, h)
        print("Object detected, initializing tracker...")
    else:
        print("No object detected. Exiting.")
        depth_stream.stop()
        cap.release()
        openni2.unload()
        sys.exit(1)

    # Initialize tracker
    tracker = cv2.TrackerKCF_create()
    tracker.init(color_img, bbox)

    print("Terminated initial object detection")

    return tracker

#################### TRACKING LOOP ####################
try:
    tracker = initial_object_detection()

    fx, fy = 525, 525
    cx, cy = 319.5, 239.5
    camera_offset_z = 100  # mm offset from gripper

    correction_factor = 0.5  # smooth factor for movement

    print("STARTING THE TRACKING LOOP")

    while True:
        # Read frames
        depth_frame = depth_stream.read_frame()
        depth_data = depth_frame.get_buffer_as_uint16()
        depth_img = np.frombuffer(depth_data, dtype=np.uint16).reshape(480, 640)
        ret, color_img = cap.read()

        success, bbox = tracker.update(color_img)

        if not success:
            print("Tracking failed!")
            break

        x, y, w, h = [int(v) for v in bbox]\
        # Take the center of the box.
        u, v = x + w // 2, y + h // 2

        # Read depth at target pixel
        # depth_value_mm = depth_img[v, u]
        depth_value_mm = read_region_depth(u, v)

        if depth_value_mm == 0:
            print("Invalid depth reading, skipping frame")
            continue

        # Compute real world coordinates
        Zc = depth_value_mm
        Xc = (u - cx) * Zc / fx
        Yc = (v - cy) * Zc / fy
        Xg, Yg, Zg = Xc, Yc, Zc - camera_offset_z

        # Compute target position
        target_x = robot_x + Xg
        target_y = robot_y + Yg
        target_z = robot_z - (Zc - 400)

        # Apply smooth correction
        current_pose = arm.get_position(is_radian=False)
        current_x, current_y, current_z = current_pose[:3]

        dx = target_x - current_x
        dy = target_y - current_y
        dz = target_z - current_z

        arm.set_position(
            x=current_x + dx * correction_factor,
            y=current_y + dy * correction_factor,
            z=current_z + dz * correction_factor,
            roll=0.0, pitch=90.0, yaw=0.0,
            speed=50, wait=True
        )

        # Visualization (optional)
        depth_display = cv2.convertScaleAbs(depth_img, alpha=0.03)
        cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Tracking", color_img)
        cv2.imshow("Depth", depth_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    depth_stream.stop()
    cap.release()
    openni2.unload()
    arm.disconnect()

except Exception as e:
    print(f"An exception occurred: {e} - {get_debug_info()}")