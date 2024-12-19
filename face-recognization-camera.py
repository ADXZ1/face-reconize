#coding=gbk
import dlib
import numpy as np
import cv2

predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
prototxt_path = 'deploy.prototxt'
model_path = 'res10_300x300_ssd_iter_140000.caffemodel'

# 初始化网络和模型
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# 特征文件路径
features_file = "face_features.txt"

# 保存特征到文件
def save_features(features):
    with open(features_file, "a") as f:
        f.write(",".join(map(str, features)) + "\n")

# 加载特征文件
def load_features():
    try:
        with open(features_file, "r") as f:
            return [np.array(list(map(float, line.strip().split(",")))) for line in f]
    except FileNotFoundError:
        return []

# 计算余弦相似度
def cosine_similarity(feature1, feature2):
    # 计算向量的点积和模长
    dot_product = np.dot(feature1, feature2)
    norm1 = np.linalg.norm(feature1)
    norm2 = np.linalg.norm(feature2)
    return dot_product / (norm1 * norm2)

# 比对特征，返回是否匹配
def compare_features(current_feature, stored_features, threshold=0.95):
    for stored_feature in stored_features:
        similarity = cosine_similarity(current_feature, stored_feature)
        if similarity > threshold:  # 相似度高于阈值则认为匹配
            return True
    return False

# 打开摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # 获取图像尺寸
    (h, w) = frame.shape[:2]

    # 预处理帧以输入到 DNN 模型
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    found_face = False
    rect = None

    # 遍历检测到的人脸
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # 调整阈值可控制检测精度
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            print(f"Face detected: startX={startX}, startY={startY}, endX={endX}, endY={endY}")
            rect = dlib.rectangle(startX, startY, endX, endY)
            found_face = True
            break

    if found_face and rect is not None:
        # 人脸关键点检测
        shape = sp(frame, rect)

        # 提取特征
        face_descriptor = facerec.compute_face_descriptor(frame, shape)
        current_feature = np.array(face_descriptor)

        # 加载已保存的特征
        stored_features = load_features()

        # 比对特征
        if compare_features(current_feature, stored_features):
            label = "Known"
        else:
            label = "Unknown"

        # 在图像上绘制矩形框和标签
        frame = cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        frame = cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 按下 's' 键保存特征值
        if label == "Unknown" and cv2.waitKey(1) & 0xFF == ord('s'):
            save_features(current_feature)
            print("Unknown face saved.")

    # 显示结果
    cv2.imshow("Face Detection", frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()