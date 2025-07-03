import os, time
import cv2
import numpy as np
from primesense import openni2

# where to dump your images
OUTPUT_DIR = "pose_camera_pictures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# init OpenNI2 & device
openni2.initialize(r"C:\Users\marcello\Downloads\OpenNI_2.3.0.86\Win64-Release\sdk\libs")
dev = openni2.Device.open_any()

# start color + depth
color_stream = dev.create_color_stream()
color_stream.start()

depth_stream = dev.create_depth_stream()
depth_stream.start()

last_save = time.time()
counter   = 0

try:
    while True:
        # ── 1) grab color ─────────────────────────────
        cf = color_stream.read_frame()
        rb = cf.get_buffer_as_triplet()
        color = np.frombuffer(rb, dtype=np.uint8).reshape((480,640,3))
        bgr   = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        # ── 2) grab depth ─────────────────────────────
        df = depth_stream.read_frame()
        db = df.get_buffer_as_uint16()
        depth = np.frombuffer(db, dtype=np.uint16).reshape((480,640))

        # normalize to 0–255 for display/save
        d8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # show live
        cv2.imshow("Color", bgr)
        cv2.imshow("Depth", d8)

        # ── 3) every 5s, save out a pair ──────────────
        if time.time() - last_save >= 5.0:
            last_save = time.time()
            fn_c = os.path.join(OUTPUT_DIR, f"color_{counter:03d}.png")
            fn_d = os.path.join(OUTPUT_DIR, f"depth_{counter:03d}.png")
            cv2.imwrite(fn_c, bgr)
            cv2.imwrite(fn_d, d8)
            print(f"[{counter:03d}] saved → {fn_c}, {fn_d}")
            counter += 1

        # quit on ‘q’
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    depth_stream.stop()
    color_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()
