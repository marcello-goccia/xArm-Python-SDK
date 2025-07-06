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
        R_base2gripper = self.rotation_from_euler(roll, pitch, yaw)

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
