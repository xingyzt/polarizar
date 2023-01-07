import cv2 as cv
import numpy as np
import math
import random
import time

with np.load('B.npz') as X:
    mtx, dist, _, _ = X

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

# define a video capture object
vid = cv.VideoCapture(3)

rho = 2  # distance resolution in pixels of the Hough grid
theta = np.pi / 360  # angular resolution in radians of the Hough grid
threshold = 50  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 50  # minimum number of pixels making up a line
max_line_gap = 100  # maximum gap in pixels between connectable line segments

objectPoints = []
realPoints = []

def randColor():
    return (random.randint(0,255),random.randint(0,255),random.randint(0,255))
  
while True:
      
    # Capture the video frame
    # by frame
    ret, img = vid.read()

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([10,120,50])
    upper = np.array([25,255,255])
    mask = cv.inRange(hsv, lower, upper)

    height,width = mask.shape
    skel = np.zeros([height,width],dtype=np.uint8)      #[height,width,3]

    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,1))

    while(np.count_nonzero(mask) != 0 ):
        eroded = cv.erode(mask,kernel)
        temp = cv.dilate(eroded,kernel)
        temp = cv.subtract(mask,temp)
        skel = cv.bitwise_or(skel,temp)
        mask = eroded.copy()

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2,2))
    skel = cv.erode(skel, kernel)
    cv.imshow("e",skel)


    lines = cv.HoughLinesP(skel, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)

    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                if y1 > y2:
                    start = (x1,y1)
                    end = (x2,y2)
                else:
                    start = (x2,y2)
                    end = (x1,y1)
                cv.line(img,(x1,y1),(x2,y2),(255,100,0), 3)
                cv.circle(img,start,4,(0,0,0),3)

    # Find the rotation and translation vectors.
    #ret, rvecs, tvecs = cv.solvePnP(realPoints, imgPoints, mtx, dist)
    # project 3D points to image plane
    #imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

    cv.imshow("res", img)
    ##time.sleep(0.5)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()

