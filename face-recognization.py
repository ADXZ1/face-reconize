#coding=gbk
import dlib
import numpy as np
import cv2

predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
prototxt_path = 'deploy.prototxt'
model_path = 'res10_300x300_ssd_iter_140000.caffemodel'

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

image_path = 'D:/face/face/00ae560f0da59645.jpg'
image = cv2.imread(image_path)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()

found_face = False
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    print(f"Detection {i}, Confidence: {confidence}")
    if confidence > 0.01:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        print(f"Face detected: startX={startX}, startY={startY}, endX={endX}, endY={endY}")
        rect = dlib.rectangle(startX, startY, endX, endY)
        found_face = True
        break
shape = sp(image, rect)
print(shape)
# 提取特征
face_descriptor = facerec.compute_face_descriptor(image, shape)#获取到128位的编码
v = np.array(face_descriptor)
print(v)

if not found_face:
    print("No face detected with sufficient confidence.")
else:
    image = cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.imshow("Detected Face", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


