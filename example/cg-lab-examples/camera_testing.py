import cv2
from primesense import openni2

# Path to your OpenNI2 Redist folder:
openni2.initialize(r"C:\Program Files\OpenNI2\Redist")  # <-- Adjust path

dev = openni2.Device.open_any()

depth_stream = dev.create_depth_stream()
depth_stream.start()

color_stream = dev.create_color_stream()
color_stream.start()

while True:
    depth_frame = depth_stream.read_frame()
    color_frame = color_stream.read_frame()

    depth_data = depth_frame.get_buffer_as_uint16()
    color_data = color_frame.get_buffer_as_uint8()

    # Convert to OpenCV image
    import numpy as np
    color_img = np.frombuffer(color_data, dtype=np.uint8).reshape(480, 640, 3)
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

    cv2.imshow("Color", color_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

depth_stream.stop()
color_stream.stop()
openni2.unload()
