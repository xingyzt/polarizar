import cv2 as cv
import numpy as np
import math
import random
import time

# colors
RED =   (0,0,255)
GREEN = (0,255,0)
YELLOW =    (0,255,255)
BLUE =  (255,0,0)
PINK =  (255,0,255)
CYAN =  (255,255,0)
WHITE = (255,255,255)

matched_new_pts = []
matched_true_pts = []

def rand_color():
    return (random.randint(0,255),random.randint(0,255),random.randint(0,255))

def rot_mat(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.float32([ [c,-s], [s,c] ])

# line segment a given by endpoint a and slope da
# line segment b given by endpoint b and slope db
# return
def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = [-da[1], da[0]] # perperdicular vector to da
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1

# Capture the webcam. Change the number if no work
vid = cv.VideoCapture(2)

cv.namedWindow("vis", cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty("vis", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

GRID_SIZE = 180
TRUE_PTS = np.float32([
             [-1,-2],          [+1,-2],
    [-2,-1], [-1,-1], [ 0,-1], [+1,-1], [+2,-1],
             [-1, 0],          [+1, 0],
    [-2,+1], [-1,+1], [ 0,+1], [+1,+1], [+2,+1],
             [-1,+2],          [+1,+2]
])
START_POS = np.float32([+1.3,+1.7])
CORNER_POS = np.float32([+3,+3])
MAP_SIZE = 700

def translate_mat(d):
    return np.float32([
       [1, 0, d[0]],
       [0, 1, d[1]]
    ])

# _position 
old_pos = START_POS
old_mat = translate_mat(START_POS)

map_img = np.zeros((MAP_SIZE,MAP_SIZE,3), np.uint8)
def map_pt(pt):
    return np.int16(MAP_SIZE/7*pt + MAP_SIZE/2)
def warped_pt(pt):
    inv = np.resize(old_mat, (3,3))
    inv[2] = np.float32([0,0,1])
    inv = np.linalg.inv(inv)
    inv.resize((2,3))
    q = (inv.dot([pt[0], pt[1], 1]))*GRID_SIZE
    return np.int16([ q[0]+X/2, q[1]+Yf ])

# Image features
matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

start_time = time.time()
while time.time() < start_time + 120:

    # Read every frame
    ret, img = vid.read()

    # Mask by hue-saturation-value
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([10,120,50])
    upper = np.array([30,255,255])
    mask = cv.inRange(hsv, lower, upper)

    # Perspective project to top-down view
    (Y, X) = img.shape[0:2]
    w = 1300
    Yf = int(Y*1.3)
    src_plane = np.float32([[0, 0], [X, 0], [X+w, Y], [-w, Y]])
    project_plane = np.float32([[0, 0], [X, 0], [X, Yf], [0, Yf]])
    project_mat = cv.getPerspectiveTransform(src_plane, project_plane)
    warped_img = cv.warpPerspective(img, project_mat, (X, Yf))

    # Find contours in mask
    new_pts = []
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:

        # Filter out small contours (likely noise)
        area = cv.contourArea(contour)
        if area < 100: continue

        # Find bounding convex hull
        hull = cv.convexHull(contour)
        #cv.drawContours(img, [hull], -1, BLUE, 1)

        # Find bounding rotated rectangle
        rect = cv.minAreaRect(hull)
        box = np.int0(cv.boxPoints(rect))
        #cv.drawContours(img, [box], -1, GREEN, 1)

        # Find centroid
        M = cv.moments(contour)
        #print(M)
        center = np.float32([
            M['m10']/M['m00'], 
            M['m01']/M['m00']
        ])

        # Fit line segment and find base of pole
        [[vx], [vy], [x], [y]] = cv.fitLine(hull, cv.DIST_L2, 0, 0.01, 0.01)
        [left, right] = sorted(box, key = lambda x:x[1])[2:]
        base = np.float32(seg_intersect(
            np.float32([x,y]), 
            np.float32([x+vx,y+vy]),
            np.float32(left), 
            np.float32(right)
        ))

        # Perspective project these features
        [warped_base, warped_left, warped_right, warped_center] = np.float32(
            cv.perspectiveTransform(np.float32([
                [base, left, right, center]
            ]), project_mat)[0]
        )

        rel_pos = np.float32([warped_base[0] - X/2, warped_base[1] - Yf])/GRID_SIZE
        new_pt = old_mat.dot([rel_pos[0], rel_pos[1], 1])
        new_pts.append(new_pt)

        cv.line(img, np.int16(base), np.int16(center), BLUE, 1)
        cv.line(warped_img, np.int16(warped_base), np.int16(warped_center), BLUE, 1)

        cv.line(img, np.int16(left), np.int16(right), BLUE, 1)
        cv.line(warped_img, np.int16(warped_left), np.int16(warped_right), BLUE, 1)

    new_pts = np.float32(new_pts)

    if len(new_pts) > 1:
        # Find similarities 
        matches = matcher.match(TRUE_PTS, new_pts)
        matches = sorted(matches, key = lambda x:x.distance)
        matches = [ x for x in matches if x.distance < 0.5 ]

        matched_true_pts = np.float32([TRUE_PTS[match.queryIdx] for match in matches])
        matched_new_pts = np.float32([new_pts[match.trainIdx] for match in matches])

        if len(matched_new_pts) > 1:

            M, inliers = cv.estimateAffinePartial2D(
                    np.array([matched_new_pts]),
                    np.array([matched_true_pts])
                )
            scale = np.linalg.det(M[:2,:2])

            M.resize((3, 3))
            M[2] = np.float32([0,0,1])

            old_mat.resize((3, 3))
            old_mat[2] = np.float32([0,0,1])

            new_mat = np.matmul(M, old_mat)

            M.resize((2, 3))
            old_mat.resize((2, 3))
            new_mat.resize((2, 3))

            new_pos = new_mat.dot([0, 0, 1])

            distance = np.linalg.norm(new_pos - old_pos)
            if distance > 1: continue # Reject sudden jumps

            #fade map img
            cv.line(map_img, map_pt(old_pos), map_pt(new_pos), rand_color(), 2)
            old_mat = np.copy(new_mat)
            old_pos = np.copy(new_pos)

    #cv.imshow("img", img)

    # map visualization
    cv.rectangle(map_img, map_pt(CORNER_POS), map_pt(-CORNER_POS), BLUE, 1)
    cv.drawMarker(map_img, map_pt(START_POS), GREEN, cv.MARKER_STAR, 20, 2)
    cv.circle(map_img, map_pt(old_pos), 1, WHITE, 2)

    tri = [
        new_mat.dot(np.float32([pt[0], pt[1], 1]))
        for pt in ([-0.1, 0.1],[0.1, 0.1],[0,-0.2])
    ]
    cv.line(map_img, map_pt(tri[0]), map_pt(tri[1]), WHITE, 2)
    cv.line(map_img, map_pt(tri[1]), map_pt(tri[2]), WHITE, 2)
    cv.line(map_img, map_pt(tri[2]), map_pt(tri[0]), WHITE, 2)

    for pt in new_pts:
        cv.drawMarker(warped_img, warped_pt(pt), RED, cv.MARKER_DIAMOND, 18, 1)
        cv.drawMarker(map_img, map_pt(pt), RED, cv.MARKER_DIAMOND, 18, 1)
    for pt in matched_new_pts:
        cv.drawMarker(warped_img, warped_pt(pt), PINK, cv.MARKER_DIAMOND, 20, 2)
        cv.drawMarker(map_img, map_pt(pt), PINK, cv.MARKER_DIAMOND, 20, 2)
    for pt in TRUE_PTS:
        cv.circle(warped_img, warped_pt(pt), 6, BLUE, 1)
        cv.circle(map_img, map_pt(pt), 6, BLUE, 1)
    for pt in matched_true_pts:
        cv.circle(warped_img, warped_pt(pt), 5, CYAN, cv.FILLED)
        cv.circle(map_img, map_pt(pt), 5, CYAN, cv.FILLED)
      
    yy = int(Yf*0.75)
    cv.rectangle(warped_img, ((X+GRID_SIZE)//2, yy), ((X-GRID_SIZE)//2, yy-GRID_SIZE), RED)

    vis = np.zeros((max(Yf, MAP_SIZE), X+MAP_SIZE,3), np.uint8)
    vis[:Yf, :X,:3] = warped_img
    vis[:MAP_SIZE, X:X+MAP_SIZE,:3] = map_img
    cv.imshow("vis", vis)

    map_img = (map_img * 0.9).astype("uint8")

    # the 'q' button is set as the
    # quitting button you may use any
    # desiRED button of your choice
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()

