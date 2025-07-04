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
import sys, os, time
import cv2
import numpy as np
from primesense import openni2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from move_robot import camera
from move_robot import robot

# allow the robot to move to at least 20 different positions, then record the poses.
MODE = "timer"     # choose "timer" or "spacebar"
time_between_images = 10

# where to dump your images
OUTPUT_DIR = "./pose_camera_pictures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

checkerboard_size = (9, 6)  # 6×6 corners interni
square_size = 20  # mm (dimensione casella)
is_camera_available = False

if not is_camera_available:
    cv2.namedWindow("Color", cv2.WINDOW_NORMAL)
    cv2.imshow("Color", np.zeros((10, 10, 3), dtype=np.uint8))  # show something small

def main():
    global camera, robot

    camera = camera.Camera()
    if is_camera_available:
        camera.initialise()
    robot = robot.Robot()

    last_save = time.time()
    counter   = 0

    try:
        print("Starting the loop:")
        while True:

            acquire_and_save_now = False
            # key = cv2.waitKey(1) & 0xFF

            if MODE == "timer":
                if time.time() - last_save >= time_between_images:
                    print(f"{time_between_images} seconds passed, acquiring image")
                    acquire_and_save_now = True
                    last_save = time.time()

            elif MODE == "spacebar":
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # spacebar pressed
                    print("spacebar pressed, acquiring image")
                    acquire_and_save_now = True
            else:
                print("Invalid MODE selected. Please choose 'timer' or 'spacebar'.")
                break

            if acquire_and_save_now:
                gray = acquire_image()
                # Find th dots of the checkboard in the pictures
                ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
                if ret:
                    acquire_robot_camera_positions(gray, corners)
                else:
                    print("Scacchiera non rilevata nella foto.")

                counter = save_images(counter, gray)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.close_streams()

def acquire_image():
    if is_camera_available:
        # ── 1) grab color ─────────────────────────────
        # Grabbing pictures of the checkboards.
        camera.read_color_frame()
        # converto grayscale
        gray = cv2.cvtColor(camera.frame_bgr, cv2.COLOR_BGR2GRAY)
        # ── 2) grab depth ─────────────────────────────
        camera.read_depth_frame()
        camera.normalise_depth_for_display()
        # show live
        cv2.imshow("Color", camera.frame_bgr)
        cv2.imshow("Depth", camera.depth_8bit)
    else:
        img = cv2.imread('checkboard.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def draw_corners_on_the_image(img, corners, ret):
    if ret:
        # Optional: disegna i corners sull’immagine
        img_corners = cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)
        cv2.imshow("Chessboard", img_corners)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def retrieve_rotations_translations_camera(gray, corners, print_on_screen=False):

    # Affina la precisione dei corners trovati
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    # Genera punti 3D reali della scacchiera (coordinate note)
    objp = np.zeros((checkerboard_size[0]*checkerboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerboard_size[0],0:checkerboard_size[1]].T.reshape(-1,2)
    objp = objp * square_size  # dimensione reale in mm

    # Calcola la posa della scacchiera rispetto alla telecamera
    retval, rvec, tvec = cv2.solvePnP(objp, corners_refined, camera.camera_matrix, camera.dist_coeffs)

    # rvec e tvec rappresentano la posa della scacchiera rispetto alla telecamera

    # (Opzionale) matrice omogenea 4x4
    R, _ = cv2.Rodrigues(rvec)
    T_cam2checkerboard = np.eye(4)
    T_cam2checkerboard[:3, :3] = R
    T_cam2checkerboard[:3, 3] = tvec.flatten()
    if print_on_screen:
        print("Rotazione (rvec):", rvec.flatten())
        print("Traslazione (tvec):", tvec.flatten())
        print("Matrice omogenea telecamera→scacchiera:\n", T_cam2checkerboard)


def acquire_robot_camera_positions(gray, corners):
    retrieve_rotations_translations_camera(gray, corners)
    _, pos_current = robot.get_robot_position()
    print("current position:", pos_current)
    _, servo_current = robot.get_robot_servo_angles()
    print("current servo", servo_current)


def save_images(counter, gray):
    # Storing the image (although not strictly necessary),
    # What is necessary here it to store the retrieved rotations, translations.
    fn_c = os.path.join(OUTPUT_DIR, f"color_{counter:03d}.png")
    fn_d = os.path.join(OUTPUT_DIR, f"depth_{counter:03d}.png")
    if is_camera_available:
        cv2.imwrite(fn_c, camera.frame_bgr)
        cv2.imwrite(fn_d, camera.depth_8bit)
    else:
        cv2.imwrite(fn_c, gray)
    print(f"[{counter:03d}] saved → {fn_c}, {fn_d}")
    counter += 1
    return counter


if __name__ == "__main__":
    main()
