import cv2
import numpy as np

# To get T (from camera to base) (eye in hand)
# place checkboard fix in the space
# move the robotin 20 different posisitons, record each pose.
# record the arm position and yaw, pitch, roll angles (arm.get_position())
arm.get_position()
# and/or the joint angles (arm.get_servo_angle())
arm.get_servo_angle()
# at the same time, take a picture of the chessboard, and extract the pose of the chessboard with respect to the camera

# Then given all this data
# Calculate the transformation matrix
# This calculation is done in openCV with the following:
cv2.calibrateHandEye()

# more code:
R_gripper2base = [...]  # rotazioni robot misurate
t_gripper2base = [...]  # traslazioni robot misurate

R_target2cam = [...]    # rotazioni camera-calibration
t_target2cam = [...]    # traslazioni camera-calibration

# Esempio di chiamata per risolvere la calibrazione Hand-Eye
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base, t_gripper2base,
    R_target2cam, t_target2cam,
    method=cv2.CALIB_HAND_EYE_TSAI
)

# Costruisci la matrice omogenea finale
T_cam2gripper = np.eye(4)
T_cam2gripper[:3, :3] = R_cam2gripper
T_cam2gripper[:3, 3] = t_cam2gripper.flatten()

# now we have the matrix T gripper to camera
# every position we can get the T gripper to base
# then we can calculate the T camera to base.


from xarm.wrapper import XArmAPI
# initialise the robot
arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)
