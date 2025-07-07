from primesense import openni2
import cv2
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from log_code import get_debug_info


class CameraMacOs:
    depth_stream = None
    color_stream = None
    depth_img = None
    frame_bgr = None  # color_image

    def __init__(self):
        self.cap = None
        self.set_camera_intrinsic_values()

    def initialise(self):
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            if not self.cap.isOpened():
                print("Error: Could not open camera.")
                raise Exception("Error: Could not open camera.")

        except Exception as e:
            print(f"An exception occurred during camera initialisation: {e} - {get_debug_info()}")

    def display_cameras(self, display_rgb=True, display_depth=True):
        """
        Break display by pressing "q"
        """
        if display_depth:
            pass
        if display_rgb:
            cv2.imshow("Color", self.frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        else:
            return True

    def close_streams(self):
        print("Closing camera streams")
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def read_region_depth(self, u, v):
        import random
        return random.randint(100, 300)

    def read_depth_frame(self):
        pass

    def normalise_depth_for_display(self):
        pass

    def read_color_frame(self):
        ret, color_frame = self.cap.read()
        if not ret:
            return False
        else:
            #self.frame_bgr = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
            self.frame_bgr = color_frame
            return True


    def initial_object_detection(self):
        # Grab frames
        try:
            print("Starting initial object detection")

            self.read_depth_frame()
            self.read_color_frame()

            hsv = cv2.cvtColor(self.frame_bgr, cv2.COLOR_BGR2HSV)

            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([179, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

            mask = mask1 + mask2

            # Call findContours first!
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                print("No object detected. Exiting.")
                return None

            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            bbox = (x, y, w, h)
            print("Object detected, initializing tracker...")

            # Initialize tracker
            tracker = cv2.TrackerCSRT_create()
            tracker.init(self.frame_bgr, bbox)

            print("Terminated initial object detection")

            return tracker
        except Exception as e:
            print(f"An exception occurred during camera acquisition: {e} - {get_debug_info()}")


    def set_camera_intrinsic_values(self):
        """
        [[ fx   0.   cx]
         [ 0.   fy   cy]
         [ 0.   0.   1.]]
        """
        self.fx, self.fy = 476.23448362, 478.26996422
        self.cx, self.cy = 318.76919937, 230.03305065
        self.camera_offset_z = 10  # mm offset from gripper
        self.correction_factor = 0.5  # smooth factor for movement

        self.camera_matrix = np.array([[self.fx,   0,        self.cx],
                                       [0,         self.fy,  self.cy],
                                       [0,         0,        1]])
        self.dist_coeffs = np.zeros(5)  # values obtained from calibration.

    def get_camera_position_world_coords(self, u, v, depth_val, debug=False):
        Xc = (u - self.cx) * depth_val / self.fx
        Yc = (v - self.cy) * depth_val / self.fy
        Zc = depth_val
        if debug:
            print(f"Xc={Xc}, Yc={Yc}, Zc={Zc}")
        return np.array([Xc, Yc, Zc, 1.0])
