import cv2
from primesense import openni2
import numpy as np


# Path to your OpenNI2 Redist folder:
openni2.initialize(r"C:\Program Files\Orbbec\OpenNI2\samples\bin")

dev = openni2.Device.open_any()

depth_stream = dev.create_depth_stream()
depth_stream.start()

cap = cv2.VideoCapture(0)

while True:
    depth_frame = depth_stream.read_frame()
    # color_frame = color_stream.read_frame()

    depth_data = depth_frame.get_buffer_as_uint16()
    # Convert buffer to numpy array
    depth_img = np.frombuffer(depth_data, dtype=np.uint16).reshape(480, 640)
    # Normalize depth for display (to uint8)
    depth_display = cv2.convertScaleAbs(depth_img, alpha=0.03)  # scaling factor depends on max range
    cv2.imshow("Depth", depth_display)

    ret, color_img = cap.read()
    cv2.imshow("Color", color_img)

    # Example of getting real-world depth:
    # Example, enter pixel
    x = 320
    y = 240
    depth_value_mm = depth_img[y, x]  # value in millimeters
    print(f"Depth at ({x},{y}) = {depth_value_mm} mm")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

depth_stream.stop()
cap.release()
openni2.unload()
