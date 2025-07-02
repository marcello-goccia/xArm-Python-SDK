from primesense import openni2
import cv2
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from log_code import get_debug_info

camera_libs_path = r"C:\Users\marcello\Downloads\OpenNI_2.3.0.86\Win64-Release\sdk\libs"


class Camera:
    depth_stream = None
    color_stream = None
    depth_img = None
    frame_bgr = None  # color_image

    def __init__(self):

        self.set_camera_intrinsic_values()

    def initialise(self):
        openni2.initialize(camera_libs_path)
        #################### CAMERA INIT ####################
        try:
            print("STARTING CAMERA INIT")

            dev = openni2.Device.open_any()

            print("Capturing depth camera")
            self.depth_stream = dev.create_depth_stream()
            self.depth_stream.start()

            print("Capturing rgb camera")
            # instead of VideoCapture
            self.color_stream = dev.create_color_stream()
            self.color_stream.start()

            # —— then try registration ——
            try:
                dev.set_image_registration_mode(
                    openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR
                )
                dev.set_depth_color_sync_enabled(True)
                print("✅ Hardware depth-to-color registration enabled")
            except Exception as e:
                print("⚠️ Registration not supported; proceeding without hardware alignment.")

            self.set_camera_intrinsic_values()

            print("CAMERA INIT - DONE")

        except Exception as e:
            print(f"An exception occurred during camera initialisation: {e} - {get_debug_info()}")

    def display_cameras(self, display_rgb=True, display_depth=True):
        """
        Break display by pressing "q"
        """
        if display_depth:
            depth_display = cv2.convertScaleAbs(self.depth_img, alpha=0.03)  # scaling factor depends on max range
            cv2.imshow("Depth", depth_display)
        if display_rgb:
            cv2.imshow("Color", self.frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        else:
            return True

    def close_streams(self):
        print("Closing camera streams")
        try:
            self.depth_stream.stop()
        except Exception:
            pass
        try:
            self.color_stream.release()
        except Exception:
            pass
        try:
            openni2.unload()
        except Exception:
            pass

    def read_region_depth(self, u, v):
        window_size = 5  # 5x5 window
        half_window = window_size // 2

        # Extract window around (u,v)
        depth_window = self.depth_img[
                       max(0, v - half_window):min(self.depth_img.shape[0], v + half_window),
                       max(0, u - half_window):min(self.depth_img.shape[1], u + half_window)
                       ]

        # Filter out zero depths
        valid_depths = depth_window[depth_window > 0]

        if valid_depths.size == 0:
            return 0

        # Use median depth (more stable)
        median_depth = float(np.median(valid_depths))
        # print(f"median depth computed: {median_depth}")
        return median_depth

    def read_depth_frame(self):
        depth_frame = self.depth_stream.read_frame()
        depth_data = depth_frame.get_buffer_as_uint16()
        self.depth_img = np.frombuffer(depth_data, dtype=np.uint16).reshape(480, 640)

    def read_color_frame(self):
        color_frame = self.color_stream.read_frame()
        color_data = color_frame.get_buffer_as_triplet()  # raw RGB bytes
        frame = np.frombuffer(color_data, dtype=np.uint8) \
            .reshape((480, 640, 3))  # (H, W, 3)

        # OpenCV expects BGR order, whereas OpenNI gives RGB:
        self.frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def initial_object_detection(self):
        # Grab frames
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
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            bbox = (x, y, w, h)
            print("Object detected, initializing tracker...")
        else:
            print("No object detected. Exiting.")
            return None

        # Initialize tracker
        tracker = cv2.TrackerCSRT_create()
        tracker.init(self.frame_bgr, bbox)

        print("Terminated initial object detection")

        return tracker

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

    def get_camera_position_world_coords(self, u, v, depth_val, debug=False):
        Xc = (u - self.cx) * depth_val / self.fx
        Yc = (v - self.cy) * depth_val / self.fy
        Zc = depth_val
        if debug:
            print(f"Xc={Xc}, Yc={Yc}, Zc={Zc}")
        return np.array([Xc, Yc, Zc, 1.0])



