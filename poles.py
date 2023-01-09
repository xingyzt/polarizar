import cv2 as cv
import numpy as np
import math
import random
import time

# colors
red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)

def rand_color():
    return (random.randint(0,255),random.randint(0,255),random.randint(0,255))

def sort_y(coords):
    return sorted(coords, key = lambda x:x[1])

def perp(a) :
    return [-a[1], a[0]]

# line segment a given by endpoint a and slope da
# line segment b given by endpoint b and slope db
# return
def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1


# Capture the webcam. Change the number if no work
vid = cv.VideoCapture(2)

calibration = np.array([736.6773965, 0., 340.95853941, 0., 736.16588227, 230.1742966,  0., 0., 1.])

# Image features
matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
oldPts = []

# Position 
oldPos = [500,500]
path = np.zeros((1000,1000,3), np.uint8)

while True:
      
    # Read every frame
    ret, img = vid.read()


    # Mask by hue-saturation-value
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([10,120,50])
    upper = np.array([25,255,255])
    mask = cv.inRange(hsv, lower, upper)

    # Perspective project to top-down view
    (Y, X) = img.shape[0:2]
    w = 1600
    srcPlane = np.float32([[0, 0], [X, 0], [X+w, Y], [-w, Y]])
    dstPlane = np.float32([[0, 0], [X, 0], [X, Y], [0, Y]])
    homographyMat = cv.getPerspectiveTransform(srcPlane, dstPlane)
    warped = cv.warpPerspective(img, homographyMat, (X, Y))

    # Find contours in mask
    newPts = []
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:

        # Filter out small contours (likely noise)
        area = cv.contourArea(contour)
        if area < 100: continue

        # Find bounding convex hull
        hull = cv.convexHull(contour)
        #cv.drawContours(img, [hull], -1, blue, 1)

        # Find bounding rotated rectangle
        rect = cv.minAreaRect(hull)
        box = np.int0(cv.boxPoints(rect))
        #cv.drawContours(img, [box], -1, green, 1)

        # Find centroid
        M = cv.moments(contour)
        center = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))

        # Fit line segment
        [[vx], [vy], [x], [y]] = cv.fitLine(hull, cv.DIST_L2, 0, 0.01, 0.01)
        [left, right] = sort_y(box)[2:]
        bottom = np.int16(seg_intersect(
            np.array([x,y]), 
            np.array([x+vx,y+vy]),
            np.array(left), 
            np.array(right))
        )

        # Perspective project these features
        [wbottom, wleft, wright, wcenter] = np.int16(cv.perspectiveTransform(np.float32([
            [bottom, left, right, center]
        ]), homographyMat)[0])

        newPts.append(wbottom)
        cv.line(warped, wbottom, wcenter, blue, 1)
        cv.line(warped, wleft, wright, blue, 1)
        cv.circle(warped, wbottom, 5, blue, 2)

    if len(oldPts) and len(newPts):
        # Find similarities 
        matches = matcher.match(np.float32(oldPts), np.float32(newPts))
        matches = sorted(matches, key = lambda x:x.distance)

        matchedOldPts = []
        matchedNewPts = []

        for match in matches:

            oldPt = oldPts[match.queryIdx]
            newPt = newPts[match.trainIdx]

            matchedOldPts.append(oldPt)
            matchedNewPts.append(newPt)

            diff = np.subtract(newPt, oldPt)
            cv.line(warped, oldPt, newPt, red, 4)
            cv.line(warped, oldPt-3*diff, newPt, red, 2)
            cv.circle(warped, oldPt, 2, red, 2)

        if len(matchedOldPts) >= 2:
            transformation, inliers = cv.estimateAffinePartial2D(
                    np.array([matchedOldPts]),
                    np.array([matchedNewPts])
                )
            newPos = transformation.dot([oldPos[0], oldPos[1], 1])
            path = (path * 0.99).astype("uint8")
            cv.line(path, np.int16(oldPos), np.int16(newPos), rand_color(), 1)
            print(newPos)
            oldPos = newPos


    if len(newPts):
        oldPts = newPts.copy()

    #cv.imshow("img", img)
    cv.imshow("img", warped)
    cv.imshow("path", path)
    #cv.moveWindow("warped", 800, 0)


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

