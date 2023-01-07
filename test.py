import cv2 as cv


img = cv.imread("test1.jpg", cv.IMREAD_COLOR)
print(img)
cv.imshow(img)
