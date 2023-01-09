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
    w = 1800
    Yf = int(Y*1.7)
    srcPlane = np.float32([[0, 0], [X, 0], [X+w, Y], [-w, Y]])
    dstPlane = np.float32([[0, 0], [X, 0], [X, Yf], [0, Yf]])
    homographyMat = cv.getPerspectiveTransform(srcPlane, dstPlane)
    warped = cv.warpPerspective(gray, homographyMat, (X, Yf))

    diff = cv.absdiff(warped, int(avg))
    
    #(minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)
    circles = cv.HoughCircles(
            diff, 
            method=cv.HOUGH_GRADIENT_ALT,
            dp=1.5,
            minDist=10,
            param1=100, # gradient threshold
            param2=0.6, # circle closeness
            minRadius=4,
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
