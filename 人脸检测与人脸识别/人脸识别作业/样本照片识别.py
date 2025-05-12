# Author: moqiHe
# Date: 2025-05-12
# Description:
#%%
#coding:utf-8
import numpy as np
import cv2 as cv
# 加载脸部特征识别器
face_cascade = cv.CascadeClassifier('./haarcascade_frontalface_alt.xml')
# 读取图片
img = cv.imread('./people.jpg')

# 将图片转化成灰度
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 设置特征的检测窗口
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv.imshow('Face Detector',img)
cv.waitKey(0)
cv.destroyAllWindows()