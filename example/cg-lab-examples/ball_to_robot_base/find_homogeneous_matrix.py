import cv2
import numpy as np

storing_poses_filename = "poses.txt"


def load_poses_from_file(filename):
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
                robot_pos = np.fromstring(lines[i+1].split(":")[1], sep=' ')
                robot_orient = np.fromstring(lines[i+2].split(":")[1], sep=' ')
                camera_rvec = np.fromstring(lines[i+3].split(":")[1], sep=' ')
                camera_tvec = np.fromstring(lines[i+4].split(":")[1], sep=' ')

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
        print(e)
        return None, None, None, None

def main():
    R_gripper2base, t_gripper2base, R_target2cam, t_target2cam = load_poses_from_file(storing_poses_filename)
    print("Done!")

if __name__ == "__main__":
    main()