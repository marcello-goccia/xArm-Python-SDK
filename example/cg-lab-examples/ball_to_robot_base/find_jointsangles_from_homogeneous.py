import cv2
import time
import numpy as np
import os, sys
from find_homogeneous_matrix import Homogeneous

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from move_robot.camera import Camera
from move_robot.cameraMacOs import CameraMacOs
from move_robot.robot import Robot
from log_code import get_debug_info

storing_poses_filename = "poses.txt"

def main():

    is_camera_available = False

    robot = Robot()
    if is_camera_available:
        camera = Camera()
    else:
        camera = CameraMacOs()

    camera.initialise()
    robot.set_initial_tracking_position()

    homogeneous = Homogeneous()

    (R_gripper2base,
     t_gripper2base,
     R_target2cam,
     t_target2cam) = homogeneous.load_poses_from_file(storing_poses_filename)

    # calibrate hand-eye
    # R_cam2gripper e t_cam2gripper formano la matrice omogenea definitiva.
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    T_cam2gripper = homogeneous.compute_final_homogeneus_matrix(R_cam2gripper, t_cam2gripper)
    T_gripper2cam = homogeneous.invert_matrix(T_cam2gripper)
    # Now we have the matrix T from gripper to camera

    # #############################
    try:
        tracker = camera.initial_object_detection()

        if tracker is None:
            camera.close_streams()
            sys.exit(1)

        print("STARTING THE TRACKING LOOP")
    except Exception as e:
        print(f"An exception occurred: {e} - {get_debug_info()}")
        camera.close_streams()

    while True:
        # 1) grab your frames
        camera.read_depth_frame()
        camera.read_color_frame()

        # 2) update tracker
        success, bbox = tracker.update(camera.frame_bgr)
        if not success:
            print("Tracker lost, trying to re-detect…")
            tracker = camera.initial_object_detection()  # or your reinit logic
            if tracker is None:
                break
            continue

        # 3) draw your bounding box
        x, y, w, h = map(int, bbox)
        cv2.rectangle(camera.frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        u, v = x + w // 2, y + h // 2

        # When the user presses 'q' the program quits.
        if not camera.display_cameras(display_rgb=True, display_depth=False):
            break

        # depth value
        z = camera.read_region_depth(u, v)
        if z == 0:
            continue

        # TRASFORMATIONS
        P_cam_hom = camera.get_camera_position_world_coords(u, v, z)

        # 4) cinematica diretta:
        _, pos_current = robot.get_robot_position()
        T_base2gripper = homogeneous.get_T_base2gripper(pos_current)

        # 5) composizione hand-eye:
        T_base2cam = T_base2gripper @ T_gripper2cam

        # 6) trasformazione punto:
        P_base_hom = T_base2cam @ P_cam_hom
        xb, yb, zb = P_base_hom[:3]

        # 7) inverse kinematics:
        roll, pitch, yaw = (0.0, 90.0, 0.0)  # es. (90,0,0)
        code, angles = robot.arm.get_ik([xb, yb, zb, roll, pitch, yaw])
        if code != 0:
            print("IK failed")
            continue

        # 8) muovi il robot
        robot.arm.set_servo_angle(angle=angles, speed=50, wait=True)

        # eventualmente un piccolo delay per stabilità
        time.sleep(1.0)

    print("Done!")

if __name__ == "__main__":
    main()