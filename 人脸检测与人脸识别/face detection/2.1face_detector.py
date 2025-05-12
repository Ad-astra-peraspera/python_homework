import numpy as np
import cv2 as cv
face_cascade = cv.CascadeClassifier('cascade_files/haarcascade_frontalface_alt.xml')
img = cv.imread('people.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()