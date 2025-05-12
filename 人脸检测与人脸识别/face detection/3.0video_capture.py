#!/usr/bin/Python
# -*- coding: utf-8 -*-
import cv2
import time
# 初始化
cap = cv2.VideoCapture(0)
# 视频绽放大小
scaling_factor = 0.5
#循环摄像，直至点击ESC键
i = 0
while True:
    # 读取一帧
    ret, frame = cap.read()
    #图像大小绽放
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor,
            interpolation=cv2.INTER_AREA)
    #显示图像
    cv2.imshow('camera', frame)
    # 获取键盘事件
    c = cv2.waitKey(1)
    if c ==ord('s'):
        i = i + 1;
        cv2.imwrite('img' + time.strftime('%m-%d-%S', time.localtime(time.time()))  + str(i) + '.jpg', frame)
    elif c == 27: #ESC
        break
# 释放摄像头
cap.release()
# 关闭窗口
cv2.destroyAllWindows()
