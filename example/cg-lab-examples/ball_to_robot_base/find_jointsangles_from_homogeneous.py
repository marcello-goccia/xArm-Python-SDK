import cv2
import numpy as np
import os, sys
from find_homogeneous_matrix import Homogeneous

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from log_code import get_debug_info

storing_poses_filename = "poses.txt"


def main():
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

    print("Done!")

if __name__ == "__main__":
    main()