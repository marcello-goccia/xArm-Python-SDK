import cv2
import numpy as np
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from log_code import get_debug_info

# storing_poses_filename = "poses.txt"

class Homogeneous:

    def __init__(self):
        pass

    def load_poses_from_file(self, filename):
        try:
            R_gripper2base = []
            t_gripper2base = []
            R_target2cam = []
            t_target2cam = []

            with open(filename, 'r') as f:
                lines = f.readlines()

            i = 0
            while i < len(lines):
                line = lines[i].strip()

                if line.startswith("# Pose"):
                    robot_pos_orient = np.fromstring(lines[i+1].split(":")[1], sep=' ')
                    robot_joints = np.fromstring(lines[i+2].split(":")[1], sep=' ')
                    camera_rvec = np.fromstring(lines[i+3].split(":")[1], sep=' ')
                    camera_tvec = np.fromstring(lines[i+4].split(":")[1], sep=' ')

                    robot_pos = robot_pos_orient[:3]  # primi 3 → XYZ
                    robot_orient = robot_pos_orient[3:]  # ultimi 3 → roll, pitch, yaw

                    # Converti orientamento robot in matrice di rotazione
                    R_gripper, _ = cv2.Rodrigues(np.deg2rad(robot_orient))
                    t_gripper = robot_pos.reshape((3,1))

                    R_cam, _ = cv2.Rodrigues(camera_rvec)
                    t_cam = camera_tvec.reshape((3,1))

                    R_gripper2base.append(R_gripper)
                    t_gripper2base.append(t_gripper)
                    R_target2cam.append(R_cam)
                    t_target2cam.append(t_cam)

                    i += 5
                else:
                    i += 1

            return R_gripper2base, t_gripper2base, R_target2cam, t_target2cam
        except Exception as e:
            print(f"{e} - {get_debug_info()}")
            return None, None, None, None

    def compute_final_homogeneus_matrix(self, R_cam2gripper, t_cam2gripper):
        # Costruisci la matrice omogenea finale
        T_cam2gripper = np.eye(4)
        T_cam2gripper[:3, :3] = R_cam2gripper
        T_cam2gripper[:3, 3] = t_cam2gripper.flatten()
        return T_cam2gripper

    def get_T_base2gripper(self, pos_current):
        # Estrai posizione e orientamento corrente
        x, y, z, roll, pitch, yaw = pos_current

        # Converti angoli da gradi a radianti
        roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])

        # Ottieni la matrice di rotazione a partire da Roll-Pitch-Yaw
        R_base2gripper = rotation_from_euler(roll, pitch, yaw)

        # Componi la matrice omogenea
        T_base2gripper = np.eye(4)
        T_base2gripper[:3, :3] = R_base2gripper
        T_base2gripper[:3, 3] = [x, y, z]

        return T_base2gripper

    def rotation_from_euler(self, roll, pitch, yaw):
        # Crea la matrice di rotazione da roll-pitch-yaw
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]
        ])
        Ry = np.array([
            [ np.cos(pitch), 0, np.sin(pitch)],
            [ 0,             1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,            0,           1]
        ])

        # La sequenza di rotazione standard è Z (yaw) → Y (pitch) → X (roll)
        R = Rz @ Ry @ Rx
        return R

    def invert_matrix(self, marix):
        return np.linalg.inv(marix)


# def main():
#
#     homogeneous = Homogeneous()
#
#     (R_gripper2base,
#      t_gripper2base,
#      R_target2cam,
#      t_target2cam) = homogeneous.load_poses_from_file(storing_poses_filename)
#
#     # calibrate hand-eye
#     # R_cam2gripper e t_cam2gripper formano la matrice omogenea definitiva.
#     R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
#         R_gripper2base, t_gripper2base,
#         R_target2cam, t_target2cam,
#         method=cv2.CALIB_HAND_EYE_TSAI
#     )
#
#     T_cam2gripper = homogeneous.compute_final_homogeneus_matrix(R_cam2gripper, t_cam2gripper)
#     T_gripper2cam = homogeneous.invert_matrix(T_cam2gripper)
#     # Now we have the matrix T from gripper to camera
#
#     ##########################################
#     # DUBBI SU QUESTO
#     ##########################################
#
#
#     # Dovrei creare un loop dove prendo posizione del robot e calcolo la trasformata BASE -> gripper
#     # ogni volta:
#     _, pos_current = robot.get_robot_position()
#     T_base2gripper = get_T_base2gripper(pos_current)
#     print("T_base2gripper:", T_base2gripper)
#
#     T_base2cam = T_base2gripper @ T_gripper2cam
#
#     ##########################################
#     # FINE DUBBI SU QUESTO
#     ##########################################
#
#     # questo punto
#     # Supponi di aver rilevato una palla rossa dalla camera, ottenendo la sua posizione nello spazio della telecamera PcamPcam
#     # Allora la posizione nello spazio della base sarà:
#     P_cam_homogeneous = np.array([x, y, z, 1])  # esempio coordinata palla in sistema telecamera
#     P_base_homogeneous = T_base2cam @ P_cam_homogeneous
#     P_base = P_base_homogeneous[:3]
#
#
#
#     # Quindi:
#     # Usare l'Inverse Kinematics per Trovare gli Angoli dei Giunti
#     # posizione XYZ calcolata precedentemente
#     x, y, z = P_base
#
#     # definire l'orientamento del gripper che desideri mantenere
#     roll, pitch, yaw = 90.0, 0.0, 0.0  # per esempio mano orizzontale
#
#     # chiedere al robot l'IK
#     code, angles = arm.get_ik([x, y, z, roll, pitch, yaw])
#
#     if code == 0:
#         print("Joint angles:", angles)
#     else:
#         print("IK failed, check your pose.")
#
#     # Finally: Muovere il Robot con gli Angoli Calcolati
#     arm.set_servo_angle(angle=angles, speed=50, wait=True)
#
#     print("Done!")
#
# if __name__ == "__main__":
#     main()