import numpy as np
import cv2 as cv
face_cascade = cv.CascadeClassifier('cascade_files/haarcascade_frontalface_alt.xml')
nose_cascade = cv.CascadeClassifier('cascade_files/haarcascade_mcs_nose.xml')
img = cv.imread('people.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    noses = nose_cascade.detectMultiScale(roi_gray,1.3,5)
    for (ex,ey,ew,eh) in noses:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()