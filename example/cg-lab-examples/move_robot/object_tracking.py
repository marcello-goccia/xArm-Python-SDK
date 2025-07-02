from camera import Camera
from robot import Robot
import sys
import cv2
import time
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from log_code import get_debug_info


def track_objects():

    is_camera_available = False

    robot = Robot()
    camera = Camera()

    if is_camera_available:
        camera.initialise()

    robot.set_initial_tracking_position()

    print(f"get_position: {robot.get_robot_position()}")
    print(f"get_servo_angle: {robot.get_robot_servo_angles()}")

    T_g2c = np.loadtxt('gripper2camera.txt')

    # #############################
    try:
        if is_camera_available:
            tracker = camera.initial_object_detection()

            if tracker is None:
                camera.close_streams()
                sys.exit(1)
        else:
            tracker = None

        print("STARTING THE TRACKING LOOP")

        while True:
            # 1) grab your frames
            if is_camera_available:
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

                # print("HERE 2 and ", success)

                # 3) draw your bounding box
                x, y, w, h = map(int, bbox)
                cv2.rectangle(camera.frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                u, v = x + w // 2, y + h // 2

                # When the user presses 'q' the program quits.
                if not camera.display_cameras(display_rgb=True, display_depth=False):
                    break

                # depth value
                depth_val = camera.read_region_depth(u, v)
                if depth_val == 0:
                    continue

            else:
                # for testing, random red ball positions
                import random as rndm
                u = rndm.randint(-200, 200)  # u should be CAMERA x (same sign y robot base green)
                v = rndm.randint(-200, 200)  # v should be CAMERA y vertical (same sign z robot base blue)
                depth_val = rndm.randint(100, 700)  # depth_val should be CAMERA depth (same sign x robot base red)
                time.sleep(2)

            # 5) compute world coords
            p_cam = camera.get_camera_position_world_coords(u, v, depth_val)
            Xc, Yc, Zc, _ = p_cam

            # 6) get current base → gripperqq
            raw = robot.get_robot_position(is_radian=False)
            if isinstance(raw, dict):
                pos = raw['pos']  # [x, y, z]
                ori = raw['ori']  # [r, p, w]
            else:
                _, arr = raw  # arr = [x, y, z, r, p, w]
                pos = arr[0:3]
                ori = arr[3:6]

            T_b2g = robot.pose_to_transform(*pos, *ori)

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
                robot.set_robot_position(x=xb, y=yb, z=zb,
                                         roll=ori[0], pitch=ori[1], yaw=ori[2],
                                         speed=50, wait=True)
            else:
                ## USE TESTING
                # adjust for gripper offset, etc...
                _, pos_current = robot.get_robot_position()

                xcurr, ycurr, zcurr, yawcurr, pitcurr, rollcurr = pos_current

                xb, yb, zb, _ = p_base

                ## where to go:
                x_new = Zc + xcurr
                y_new = Yc + ycurr
                z_new = Xc + zcurr

                camera_x = u  # -18
                camera_y = v  # -80
                camera_depth = 200  # depth_val
                offset_camera_height = 300
                offset_camera_depth = 200
                robot.set_robot_position(x=camera_depth + offset_camera_depth,
                                         y=-camera_x,
                                         z=camera_y + offset_camera_height,
                                         speed=100,
                                         wait=True)  ## x is depth, y is base x, z is base y

        # end loop

        camera.close_streams()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An exception occurred: {e} - {get_debug_info()}")
        camera.close_streams()
        cv2.destroyAllWindows()



def main():
    track_objects()

if __name__ == "__main__":
    main()