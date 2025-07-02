"""
•	Posiziona un riferimento noto (es. scacchiera (checkerboard) fisso nello spazio, visibile dalla camera.
•	Muovi il robot in almeno 10-20 pose diverse, registrando ad ogni posa:
        - La posa del robot rispetto alla sua base (che puoi leggere con arm.get_position() e arm.get_orientation()
        oppure direttamente gli angoli delle giunture con arm.get_servo_angle()).
        - Contemporaneamente, scatti una foto della scacchiera/marker e ***** ne estrai la posa rispetto alla telecamera *****
        (usando OpenCV).

Quindi avrai ad esempio:

Numero	Robot base→End effector	    Camera→Checkerboard
1	    nota	                    nota
2	    nota	                    nota
...	...	...

"""


import cv2
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from move_robot import camera
from move_robot import robot

camera = camera.Camera()
robot = robot.Robot()

# Parametri intrinseci già calcolati (esempio)
camera_matrix = np.array([[camera.fx, 0,         camera.cx],
                          [0,         camera.fy, camera.cy],
                          [0,         0,          1]])
dist_coeffs = np.zeros(5)  # o i valori ottenuti dalla calibrazione

# Foto della scacchiera scattata dalla camera robot
img = cv2.imread('checkboard.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Impostazioni della scacchiera
checkerboard_size = (9, 6)  # 6×6 corners interni
square_size = 20  # mm (dimensione casella)

# Trova i punti della scacchiera nella foto
ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

# if ret:
#     # Optional: disegna i corners sull’immagine
#     img_corners = cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)
#     cv2.imshow("Chessboard", img_corners)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

if ret:
    # Affina la precisione dei corners trovati
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    # Genera punti 3D reali della scacchiera (coordinate note)
    objp = np.zeros((checkerboard_size[0]*checkerboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerboard_size[0],0:checkerboard_size[1]].T.reshape(-1,2)
    objp = objp * square_size  # dimensione reale in mm

    # Calcola la posa della scacchiera rispetto alla telecamera
    retval, rvec, tvec = cv2.solvePnP(objp, corners_refined, camera_matrix, dist_coeffs)

    # rvec e tvec rappresentano la posa della scacchiera rispetto alla telecamera
    print("Rotazione (rvec):", rvec.flatten())
    print("Traslazione (tvec):", tvec.flatten())

    # (Opzionale) matrice omogenea 4x4
    R, _ = cv2.Rodrigues(rvec)
    T_cam2checkerboard = np.eye(4)
    T_cam2checkerboard[:3, :3] = R
    T_cam2checkerboard[:3, 3] = tvec.flatten()
    print("Matrice omogenea telecamera→scacchiera:\n", T_cam2checkerboard)

else:
    print("Scacchiera non rilevata nella foto.")

