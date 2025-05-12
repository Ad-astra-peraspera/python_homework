# Author: moqiHe
# Date: 2025-05-12
# Description:
#%%
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
            label_words.append(os.path.basename(os.path.dirname(filepath)))

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
            name = os.path.basename(os.path.dirname(filepath))

            # 检测是否有人脸，并获得人脸数据
            faces = face_cascade.detectMultiScale(image, 1.1, 2, minSize=(100, 100))

            # 输入每个人脸
            for (x, y, w, h) in faces:
                images.append(image[y:y + h, x:x + w])
                labels.append(le.word_to_num(name))

    return images, labels, le


if __name__ == '__main__':
    cascade_path = "haarcascade_frontalface_alt.xml"
    path_train = './faces_dataset/train'
    path_test = './faces_dataset/test'

    # 人脸检测训练结果读取
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # 人脸识别方法初始化
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # 获取训练数据集
    images, labels, le = get_images_and_labels(path_train)

    # 读取训练模型
    print (u'\n读取模型...')
    recognizer.read('my_LBPHFaceRecognizer.xml')

    # 识别测试数据集
    print (u'\n识别视频中的人脸...')

    cap = cv2.VideoCapture(0)
    # Define the scaling factor
    scaling_factor = 0.8

    # 点击ESC退出
    while True:
        # Capture the current frame and resize it
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor,
                           interpolation=cv2.INTER_AREA)


        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Run the face detector on the grayscale image
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
        predicted_person = ''
        # Draw rectangles on the image
        for (x, y, w, h) in face_rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            predicted_index, conf = recognizer.predict(
                gray[y:y + h, x:x + w])

            # 文字到数字的转换
            predicted_person = le.num_to_word(predicted_index)

            cv2.rectangle(frame, (x, y + h - 30), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame[y:y + h, x:x + w], predicted_person,
                    (6,h - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        # Display the image
        cv2.imshow("result", frame)
        # Check if Esc key has been pressed
        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
