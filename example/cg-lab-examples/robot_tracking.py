import cv2
from primesense import openni2
import numpy as np
import math
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

depth_stream = None
color_stream = None

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

    T_g2c = np.loadtxt('gripper2camera.txt')

    # —— then try registration ——
    try:
        dev.set_image_registration_mode(
            openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR
        )
        dev.set_depth_color_sync_enabled(True)
        print("✅ Hardware depth-to-color registration enabled")
    except Exception as e:
        print("⚠️ Registration not supported; proceeding without hardware alignment.")


    print("Move robot to starting point")
    robot_x, robot_y, robot_z = 266.0, 0.0, 303.0
    roll, pitch, yaw = -126,-88, -55
    # arm.set_position(x=robot_x, y=robot_y, z=robot_z, roll=0.0, pitch=90.0, yaw=0.0, speed=150, wait=True)
    arm.set_position(x=robot_x, y=robot_y, z=robot_z, roll=roll, pitch=pitch, yaw=yaw, speed=100, wait=True)
    time.sleep(2)

    print("CAMERA INIT - DONE")

except Exception as e:
    print(f"An exception occurred during camera initialisation: {e} - {get_debug_info()}")



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

        # adjust for gripper offset, etc...
        xb, yb, zb, _ = p_base

        # uncomment top make the arm move.
        # arm.set_position(
        #         x = xb, y = yb, z = zb,
        #         roll = ori[0], pitch = ori[1], yaw = ori[2],
        #         speed = 50, wait = True)
        arm.set_position(
                x = Yc, y = Xc, z = robot_z,
                # x = robot_x + Xc, y = robot_y + Yc, z = robot_z + Zc - 600,
                # roll = -126, pitch = -88, yaw = -55,
                speed = 50, wait = True)
        # robot_x, robot_y, robot_z = 266.0, 0.0, 303.0

    # end loop

    close_streams(depth_stream, color_stream, openni2, arm)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"An exception occurred: {e} - {get_debug_info()}")
    close_streams(depth_stream, color_stream, openni2, arm)
    cv2.destroyAllWindows()
