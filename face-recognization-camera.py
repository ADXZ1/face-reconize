#coding=gbk
import dlib
import numpy as np
import cv2

predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
prototxt_path = 'deploy.prototxt'
model_path = 'res10_300x300_ssd_iter_140000.caffemodel'

# ��ʼ�������ģ��
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# �����ļ�·��
features_file = "face_features.txt"

# �����������ļ�
def save_features(features):
    with open(features_file, "a") as f:
        f.write(",".join(map(str, features)) + "\n")

# ���������ļ�
def load_features():
    try:
        with open(features_file, "r") as f:
            return [np.array(list(map(float, line.strip().split(",")))) for line in f]
    except FileNotFoundError:
        return []

# �����������ƶ�
def cosine_similarity(feature1, feature2):
    # ���������ĵ����ģ��
    dot_product = np.dot(feature1, feature2)
    norm1 = np.linalg.norm(feature1)
    norm2 = np.linalg.norm(feature2)
    return dot_product / (norm1 * norm2)

# �ȶ������������Ƿ�ƥ��
def compare_features(current_feature, stored_features, threshold=0.95):
    for stored_feature in stored_features:
        similarity = cosine_similarity(current_feature, stored_feature)
        if similarity > threshold:  # ���ƶȸ�����ֵ����Ϊƥ��
            return True
    return False

# ������ͷ
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # ��ȡͼ��ߴ�
    (h, w) = frame.shape[:2]

    # Ԥ����֡�����뵽 DNN ģ��
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    found_face = False
    rect = None

    # ������⵽������
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # ������ֵ�ɿ��Ƽ�⾫��
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            print(f"Face detected: startX={startX}, startY={startY}, endX={endX}, endY={endY}")
            rect = dlib.rectangle(startX, startY, endX, endY)
            found_face = True
            break

    if found_face and rect is not None:
        # �����ؼ�����
        shape = sp(frame, rect)

        # ��ȡ����
        face_descriptor = facerec.compute_face_descriptor(frame, shape)
        current_feature = np.array(face_descriptor)

        # �����ѱ��������
        stored_features = load_features()

        # �ȶ�����
        if compare_features(current_feature, stored_features):
            label = "Known"
        else:
            label = "Unknown"

        # ��ͼ���ϻ��ƾ��ο�ͱ�ǩ
        frame = cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        frame = cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # ���� 's' ����������ֵ
        if label == "Unknown" and cv2.waitKey(1) & 0xFF == ord('s'):
            save_features(current_feature)
            print("Unknown face saved.")

    # ��ʾ���
    cv2.imshow("Face Detection", frame)

    # ���� 'q' ���˳�
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# �ͷ���Դ
cap.release()
cv2.destroyAllWindows()