import cv2 as cv
import numpy as np

# define a video capture object
vid = cv.VideoCapture(2)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, img = vid.read()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)

    cv.circle(img, minLoc, 50, (0, 255, 0), 2)

    cv.imshow('detected circles',img)

      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()

