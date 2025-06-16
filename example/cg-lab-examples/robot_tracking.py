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
    robot_x, robot_y, robot_z = 300.0, 200.0, 400.0
    arm.set_position(x=robot_x, y=robot_y, z=robot_z, roll=0.0, pitch=90.0, yaw=0.0, speed=150, wait=True)
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
    print(f"median depth computed: {median_depth}")
    return median_depth



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

        # 2) update tracker
        success, bbox = tracker.update(frame_bgr)
        if not success:
            print("Tracker lost, trying to re-detect…")
            tracker = initial_object_detection()  # or your reinit logic
            if tracker is None:
                break
            continue

        # 3) draw your bounding box
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        u, v = x + w // 2, y + h // 2

        # 4) display
        depth_display = cv2.convertScaleAbs(depth_img, alpha=0.03)
        cv2.imshow("Tracker", frame_bgr)
        #cv2.imshow("Depth", depth_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 5) compute world coords & move robot
        depth_val = read_region_depth(depth_img, u, v)
        if depth_val == 0:
            continue

        Zc = depth_val
        Xc = (u - cx) * Zc / fx
        Yc = (v - cy) * Zc / fy
        # adjust for gripper offset, etc...
        target_x = robot_x + Xc
        target_y = robot_y + Yc
        target_z = robot_z - (Zc - 400)

        # get current pose
        raw = arm.get_position(is_radian=False)
        if isinstance(raw, dict):
            cp = raw['pos']
        else:
            cp = raw[1]
        cx_, cy_, cz_ = cp[:3]

        # interpolate & send new target
        arm.set_position(
            x=cx_ + (target_x - cx_) * correction_factor,
            y=cy_ + (target_y - cy_) * correction_factor,
            z=cz_ + (target_z - cz_) * correction_factor,
            roll=0.0, pitch=90.0, yaw=0.0,
            speed=50, wait=True
        )

    # end loop

    close_streams(depth_stream, color_stream, openni2, arm)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"An exception occurred: {e} - {get_debug_info()}")
    close_streams(depth_stream, color_stream, openni2, arm)
    cv2.destroyAllWindows()
