import glob
import numpy as np
import cv2

# termination criteria for cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001)

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)  # adjust to your board

objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('color_calib/*.png')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # cv2.drawChessboardCorners(img, (9,6), corners2, ret)
        # cv2.imshow('img', img); cv2.waitKey(100)
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

print("Camera matrix:", mtx)
print("Dist coeffs:", dist.ravel())
