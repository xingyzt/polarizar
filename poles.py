import cv2 as cv
import numpy as np
import math
import random
import time

def randColor():
    return (random.randint(0,255),random.randint(0,255),random.randint(0,255))

# Capture the webcam. Change the number if no work
vid = cv.VideoCapture(0)

# Image features
oldFeatures = []
  
while True:
      
    # Read every frame
    ret, img = vid.read()

    # Warp to top-down view
    (Y, X) = img.shape[0:2]
    srcPlane = np.float32([[0, 0], [X, 0], [X+1000, Y], [-1000, Y]])
    dstPlane = np.float32([[0, 0], [X, 0], [X, Y], [0, Y]])
    homographyMat = cv.getPerspectiveTransform(srcPlane, dstPlane)
    warped = cv.warpPerspective(img, homographyMat, (X, Y))

    # Height & width of image
    rows, cols = img.shape[:2]

    # Mask by hue-saturation-value
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([10,120,50])
    upper = np.array([25,255,255])
    mask = cv.inRange(hsv, lower, upper)

    # Find contours in mask
    newFeatures = []
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:

        # Filter out small contours (likely noise)
        area = cv.contourArea(contour)
        if area < 100: continue

        # Find bounding convex hull
        hull = cv.convexHull(contour)
        cv.drawContours(img, [hull], -1, (255,0,0), 1)

        # Find bounding rotated rectangle
        rect = cv.minAreaRect(hull)
        box = np.int0(cv.boxPoints(rect))
        cv.drawContours(img, [box], -1, (0,255,0), 1)

        # Find centroid
        M = cv.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        # Fit line segment
        perimeter = cv.arcLength(hull, True)
        [vx, vy, x, y] = cv.fitLine(hull, cv.DIST_L2, 0, 0.01, 0.01)
        magnitude = perimeter / 4 / pow(vx*vx + vy*vy, 0.5)
        dx = int(magnitude*vx*np.sign(vy))
        dy = int(magnitude*abs(vy))
        top = (cx-dx, cy-dy)
        bottom = (cx+dx, cy+dy)

        cv.line(img, bottom, top, (0,0,255), 3)
        cv.circle(img, bottom, 10, (255,0,0), 3)

        [wbottom, wtop] = np.int16(cv.perspectiveTransform(np.float32([[bottom, top]]), homographyMat)[0])
        newFeatures.append(wbottom)
        cv.line(warped, wbottom, wtop, (0,0,255), 3)
        cv.circle(warped, wbottom, 10, (255,0,0), 3)

    cv.imshow("img", img)
    cv.imshow("warped", warped)
    cv.moveWindow("warped", 800, 0)
    oldFeatures = newFeatures.copy()



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

