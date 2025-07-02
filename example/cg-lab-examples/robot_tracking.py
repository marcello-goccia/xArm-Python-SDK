import cv2
from primesense import openni2
import numpy as np
import math
import os
import sys
import time
from log_code import get_debug_info

# DOCKER UFACTORY SIMULATOR
# from https://forum.ufactory.cc/t/ufactory-studio-simulation/3719
# TO CREATE THE CONTAINER
# docker run -it --name uf_software -p 18333:18333 -p 502:502 -p 503:503 -p 504:504 -p 30000:30000 -p 30001:30001 -p 30002:30002 -p 30003:30003 danielwang123321/uf-ubuntu-docker
# TO RUN EXISTING CONTAINER
# docker exec -it uf_software -p 18333:18333 -p 502:502 -p 503:503 -p 504:504 -p 30000:30000 -p 30001:30001 -p 30002:30002 -p 30003:30003 danielwang123321/uf-ubuntu-docker
# THEN
# /xarm_scripts/xarm_start.sh 7 7
# Run a web browser and input 127.0.0.1:18333 or locathost:18333
# OR run ufactory aplication and input 127.0.0.1:18333 or locathost:18333

is_camera_available = False

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
time.sleep(1)

print("ROBOT INIT - DONE")

depth_stream = None
color_stream = None

if is_camera_available:
    #################### CAMERA INIT ####################
    try:
        print("STARTING CAMERA INIT")

        openni2.initialize(r"C:\Users\marcello\Downloads\OpenNI_2.3.0.86\Win64-Release\sdk\libs")
        dev = openni2.Device.open_any()

        print("Capturing depth camera")
        depth_stream = dev.create_depth_stream()
        depth_stream.start()

        print("Capturing rgb camera")
        # instead of VideoCapture
        color_stream = dev.create_color_stream()
        color_stream.start()

        # —— then try registration ——
        try:
            dev.set_image_registration_mode(
                openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR
            )
            dev.set_depth_color_sync_enabled(True)
            print("✅ Hardware depth-to-color registration enabled")
        except Exception as e:
            print("⚠️ Registration not supported; proceeding without hardware alignment.")

        print("CAMERA INIT - DONE")

    except Exception as e:
        print(f"An exception occurred during camera initialisation: {e} - {get_debug_info()}")


print("Move robot to starting point")
robot_x, robot_y, robot_z = 266.0, 0.0, 303.0
roll, pitch, yaw = -126,-88, -55
arm.set_position(x=robot_x, y=robot_y, z=robot_z, roll=roll, pitch=pitch, yaw=yaw, speed=100, wait=True)
time.sleep(2)

print(f"get_position: {arm.get_position()}")

print(f"get_servo_angle: {arm.get_servo_angle()}")

T_g2c = np.loadtxt('gripper2camera.txt')

# arm.set_position(x=300, y=-14, z=-80, speed=100, wait=True)
#
#
# camera_x = -200
# camera_y = 100
# camera_depth = 500
# arm.set_position(x=camera_depth, y=-camera_x, z=camera_y + offset_camera_height, speed=100, wait=True)  ## x is depth, y is base x, z is base y
# time.sleep(3)

# # arm.set_position(x=300, y=-14, z=-80, speed=100, wait=True)

# import random as rndm
# while True:
#     camera_x = rndm.randint(-200, 200)
#     camera_y = rndm.randint(-200, 200)
#     camera_depth = 500
#     print(f"camera_x: {camera_x}, camera_y: {camera_y}")
#     arm.set_position(x=camera_depth, y=-camera_x, z=camera_y + offset_camera_height, roll=90, pitch=0, yaw=90,
#                      speed=100, wait=True)  ## x is depth, y is base x, z is base y
#     time.sleep(3)


def display_cameras(depth_img, color_img, display=False):
    if display:
        depth_display = cv2.convertScaleAbs(depth_img, alpha=0.03)  # scaling factor depends on max range
        cv2.imshow("Depth", depth_display)
        cv2.imshow("Color", color_img)


def close_streams(depth_stream, color_stream , openni2, arm):
    print("Closing camera streams")
    try:
        depth_stream.stop()
    except Exception:
        pass
    try:
        color_stream .release()
    except Exception:
        pass
    try:
        openni2.unload()
    except Exception:
        pass
    try:
        arm.disconnect()
    except Exception:
        pass


def read_region_depth(depth_img, u, v):
    window_size = 5  # 5x5 window
    half_window = window_size // 2

    # Extract window around (u,v)
    depth_window = depth_img[
                   max(0, v - half_window):min(depth_img.shape[0], v + half_window),
                   max(0, u - half_window):min(depth_img.shape[1], u + half_window)
                   ]

    # Filter out zero depths
    valid_depths = depth_window[depth_window > 0]

    if valid_depths.size == 0:
        return 0

    # Use median depth (more stable)
    median_depth = float(np.median(valid_depths))
    # print(f"median depth computed: {median_depth}")
    return median_depth


def pose_to_transform(x, y, z, roll, pitch, yaw):
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

#################### INITIAL OBJECT DETECTION ####################
def initial_object_detection():
    # Grab frames
    print("Starting initial object detection")

    depth_frame = depth_stream.read_frame()
    depth_data = depth_frame.get_buffer_as_uint16()
    depth_img = np.frombuffer(depth_data, dtype=np.uint16).reshape(480, 640)

    color_frame = color_stream.read_frame()
    color_data = color_frame.get_buffer_as_triplet()  # raw RGB bytes
    frame = np.frombuffer(color_data, dtype=np.uint8) \
        .reshape((480, 640, 3))  # (H, W, 3)
    # OpenCV expects BGR order, whereas OpenNI gives RGB:
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

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
        return None

    # Initialize tracker
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame_bgr, bbox)

    print("Terminated initial object detection")

    return tracker


#################### TRACKING LOOP ####################
try:
    if is_camera_available:
        tracker = initial_object_detection()

        if tracker is None:
            close_streams(depth_stream, color_stream , openni2, arm)
            sys.exit(1)

    """
    [[476.23448362   0.         318.76919937]
     [  0.         478.26996422 230.03305065]
     [  0.           0.           1.        ]]
    """

    fx, fy = 476.23448362, 478.26996422
    cx, cy = 318.76919937, 230.03305065
    camera_offset_z = 10  # mm offset from gripper

    correction_factor = 0.5  # smooth factor for movement

    print("STARTING THE TRACKING LOOP")

    while True:
        # 1) grab your frames
        if is_camera_available:
            depth_frame = depth_stream.read_frame()
            depth_data = depth_frame.get_buffer_as_uint16()
            depth_img = np.frombuffer(depth_data, dtype=np.uint16).reshape(480, 640)

            color_frame = color_stream.read_frame()
            rgb_data = color_frame.get_buffer_as_triplet()
            frame_rgb = np.frombuffer(rgb_data, dtype=np.uint8).reshape((480, 640, 3))
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            #print("HERE 1")

            # 2) update tracker
            success, bbox = tracker.update(frame_bgr)
            if not success:
                print("Tracker lost, trying to re-detect…")
                tracker = initial_object_detection()  # or your reinit logic
                if tracker is None:
                    break
                continue

            #print("HERE 2 and ", success)

            # 3) draw your bounding box
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            u, v = x + w // 2, y + h // 2

            #print("HERE 3 and ", u, v)

            # 4) display
            depth_display = cv2.convertScaleAbs(depth_img, alpha=0.03)
            cv2.imshow("Tracker", frame_bgr)
            #cv2.imshow("Depth", depth_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            #print("HERE 4 and ", depth_display)

            # 5) compute world coords & move robot
            depth_val = read_region_depth(depth_img, u, v)
            if depth_val == 0:
                continue

            # print("HERE 5 and ", depth_val)

        else:
            import random as rndm
            u = rndm.randint(-200, 200)  # u should be CAMERA x (same sign y robot base green)
            v = rndm.randint(-200, 200)  # v should be CAMERA y vertical (same sign z robot base blue)
            depth_val = rndm.randint(100, 700)  # depth_val should be CAMERA depth (same sign x robot base red)
            time.sleep(2)

        Xc = (u - cx) * depth_val  / fx
        Yc = (v - cy) * depth_val  / fy
        Zc = depth_val
        print(f"Xc={Xc}, Yc={Yc}, Zc={Zc}")
        p_cam = np.array([Xc, Yc, Zc, 1.0])

        # 6) get current base→gripperqq
        raw = arm.get_position(is_radian=False)
        if isinstance(raw, dict):
            pos = raw['pos']  # [x, y, z]
            ori = raw['ori']  # [r, p, w]
        else:
            _, arr = raw  # arr = [x, y, z, r, p, w]
            pos = arr[0:3]
            ori = arr[3:6]
        T_b2g = pose_to_transform(*pos, *ori)

        # 3) ball‐in‐base
        p_base = T_b2g.dot(T_g2c).dot(p_cam)
        xb, yb, zb, _ = p_base

        # 4) optional: if you want the camera to sit 10 mm *above* the ball
        #    in the camera’s optical Z (so you don’t collide)
        p_cam_target = np.array([0, 0, Zc - 10.0, 1.0])
        # what gripper‐in‐base would put the camera there?
        T_b2g_target = T_b2g.dot(
            np.linalg.inv(T_g2c)
        ).dot(p_cam_target.reshape(4, 1))
        xt, yt, zt = T_b2g_target[0:3, 0:1].flatten()

        normal_use = False
        if normal_use:
            ### NORMAL USE
            arm.set_position(
                    x = xb, y = yb, z = zb,
                    roll = ori[0], pitch = ori[1], yaw = ori[2],
                    speed = 50, wait = True)
        else:
            ## USE TESTING
            # adjust for gripper offset, etc...
            _, pos_current = arm.get_position()

            xcurr, ycurr, zcurr, yawcurr, pitcurr, rollcurr = pos_current

            xb, yb, zb, _ = p_base

            ## where to go:
            x_new = Zc + xcurr
            y_new = Yc + ycurr
            z_new = Xc + zcurr

            camera_x = u  #-18
            camera_y = v  #-80
            camera_depth = 200  # depth_val
            offset_camera_height = 300
            offset_camera_depth = 200
            arm.set_position(x=camera_depth + offset_camera_depth, y=-camera_x, z=camera_y + offset_camera_height, speed=100,
                             wait=True)  ## x is depth, y is base x, z is base y

    # end loop

    close_streams(depth_stream, color_stream, openni2, arm)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"An exception occurred: {e} - {get_debug_info()}")
    close_streams(depth_stream, color_stream, openni2, arm)
    cv2.destroyAllWindows()
