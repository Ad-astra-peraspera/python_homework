#!/usr/bin/Python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from sklearn import preprocessing

# 为图像做标记，并将文字转化为数字，再训练
class LabelEncoder(object):
    # 编码：文字到数字
    def encode_labels(self, label_words):
        self.le = preprocessing.LabelEncoder()
        self.le.fit(label_words)

    # 数字转换成文字
    def word_to_num(self, label_word):
        return int(self.le.transform([label_word])[0])

    # 数学到文字的转换
    def num_to_word(self, label_num):
        return self.le.inverse_transform([label_num])[0]

# 根据路径获取图片
def get_images_and_labels(input_path):
    label_words = []

    # 循环读取所有图片
    for root, dirs, files in os.walk(input_path):
        for filename in (x for x in files if x.endswith('.jpg')):
            filepath = os.path.join(root, filename)
            label_words.append(filepath.split('\\')[-2])
            
    # 编码
    images = []
    le = LabelEncoder()
    le.encode_labels(label_words)
    labels = []

    # Parse the input directory
    for root, dirs, files in os.walk(input_path):
        for filename in (x for x in files if x.endswith('.jpg')):
            filepath = os.path.join(root, filename)

            # 读入灰度图
            image = cv2.imread(filepath, 0)
            # 获取标记
            name = filepath.split('\\')[-2]
            # 检测是否有人脸，并获得人脸数据
            faces = faceCascade.detectMultiScale(image, 1.1, 2, minSize=(100,100))
            # 输入每个人脸
            for (x, y, w, h) in faces:
                images.append(image[y:y+h, x:x+w])
                labels.append(le.word_to_num(name))

    return images, labels, le

if __name__=='__main__':
    cascade_path = "cascade_files/haarcascade_frontalface_alt.xml"
    path_train = 'faces_dataset/train'
    path_test = 'faces_dataset/test'

    # 人脸检测训练结果读取
    faceCascade = cv2.CascadeClassifier(cascade_path)
    # 人脸识别方法初始化
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # 获取训练数据集
    images, labels, le = get_images_and_labels(path_train)
    # 模型训练
    print(u"\n使用训练集对模型进行训练...")
    recognizer.train(images, np.array(labels))
    #保存训练模型
    recognizer.save('my_LBPHFaceRecognizer.xml')

    # 识别测试数据集
    print(u'\n识别图像中的人脸...')
    stop_flag = False
    for root, dirs, files in os.walk(path_test):
        for filename in (x for x in files if x.endswith('.jpg')):
            filepath = os.path.join(root, filename)
            predict_image = cv2.imread(filepath,0)
            # 人脸检测
            faces = faceCascade.detectMultiScale(predict_image, 1.1, 
                    2, minSize=(100,100))

            # 人脸识别
            for (x, y, w, h) in faces:

                # 识别
                predicted_index, conf = recognizer.predict(
                        predict_image[y:y+h, x:x+w])
                print(predicted_index)
                # 文字到数字的转换
                predicted_person = le.num_to_word(predicted_index)

                # 显示结果（彩色）
                predict_image = cv2.imread(filepath)
                cv2.rectangle(predict_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(predict_image, (x, y + h - 35), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(predict_image, predicted_person,
                        (x + 6,y + h - 6),cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255), 2)
                cv2.imshow("result", predict_image)
                cv2.imwrite("result-" + predicted_person + '.jpg', predict_image)

                cv2.waitKey(0)
                stop_flag = True
                break

        if stop_flag:
            break

