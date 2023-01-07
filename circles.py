import cv2 as cv
import numpy as np

# define a video capture object
vid = cv.VideoCapture(2)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, img = vid.read()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    avg = np.average(np.average(gray, axis=0), axis=0)

    (Y, X) = img.shape[0:2]
    X-=1
    Y-=1
    W=0
    H=0
    w=1000
    
    srcPlane = np.array([[W, H], [X-W, H], [X+w, Y], [-w, Y]])
    dstPlane = np.array([[0,0], [X, 0], [X, Y], [0, Y]])

    pts = srcPlane.reshape((-1,1,2))
    cv.polylines(img,[pts],True,(0,255,255))

    cv.imshow("un",img)
    homographyMat, status = cv.findHomography(srcPlane, dstPlane)
    warped = cv.warpPerspective(gray, homographyMat, (X, Y))
    warped = cv.resize(warped, (int(warped.shape[0]*1.2), warped.shape[1]))

    diff = cv.absdiff(warped, int(avg))
    
    #(minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)
    circles = cv.HoughCircles(
            diff, 
            method=cv.HOUGH_GRADIENT_ALT,
            dp=1.5,
            minDist=100,
            param1=10, # gradient threshold
            param2=0.6, # circle closeness
            minRadius=8,
            maxRadius=12
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv.circle(diff,(i[0],i[1]),i[2],255,1)
            # draw the center of the circle
            cv.circle(diff,(i[0],i[1]),2,255,3)

    cv.imshow('detected circles',diff)

      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()
