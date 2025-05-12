# Author: moqiHe
# Date: 2025-05-12
# Description:
import cv2
import os

# 加载训练好的模型
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('my_LBPHFaceRecognizer.xml')

# 加载 Haar 人脸检测模型
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# 指定测试图片路径
test_img_path = 'faces_dataset/test/b15.jpg'  # 确保这张图片在同级目录或写绝对路径
img = cv2.imread(test_img_path)
if img is None:
    raise FileNotFoundError("无法加载测试图片，请检查路径是否正确")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 检测人脸
faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# 构造标签（需和训练时一致）
label_dict = {
    0: 'elder brother',
    1: 'father',
    2: 'mother',
    3: 'youger brother'  # 注意拼写是 youger，不是 younger
}

for (x, y, w, h) in faces:
    roi = gray[y:y+h, x:x+w]
    label_id, confidence = recognizer.predict(roi)
    predicted_person = label_dict.get(label_id, "unknown")


    predict_image = cv2.imread('faces_dataset/test/b15.jpg')
    cv2.rectangle(predict_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(predict_image, (x, y + h - 35), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(predict_image, predicted_person,
                (x + 6, y + h - 6), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)
    cv2.imshow("result", predict_image)
    cv2.imwrite("result-" + predicted_person + '.jpg', predict_image)


    # 显示图像
    cv2.imshow("识别结果", img)
    cv2.waitKey(0)



cv2.destroyAllWindows()