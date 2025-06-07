import os
import numpy as np
import csv
import cv2
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import mediapipe as mp

train_dir = r"data\train_augmented"
output_csv = r"stacking_train_features.csv"

model_cnn = load_model("weights/cnn_mlp_model.h5")
cnn_labels = joblib.load("weights/cnn_mlp_labels.joblib")
label_list = [k for k, v in sorted(cnn_labels.items(), key=lambda x: x[1])]
model_mp = load_model("weights_mediapipe/mediapipe_mlp_model.h5")


cnn_feat_model = MobileNetV2(include_top=False, pooling='avg', input_shape=(224, 224, 3))
cnn_feat_model.trainable = False

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)


def extract_cnn_feature(img):
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img.astype(np.float32))
    return cnn_feat_model.predict(np.expand_dims(img, axis=0), verbose=0)[0]

def extract_keypoints(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0].landmark
        return np.array([[p.x, p.y] for p in lm]).flatten()
    return None

print("正在提取训练集特征……")
features = []

header = [f"cnn_{c}" for c in label_list] + [f"mp_{c}" for c in label_list] + ["label"]

for class_name in sorted(os.listdir(train_dir)):
    class_path = os.path.join(train_dir, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        cnn_feat = extract_cnn_feature(img)
        cnn_pred = model_cnn.predict(np.expand_dims(cnn_feat, axis=0), verbose=0)[0]
        keypoints = extract_keypoints(img)
        if keypoints is not None and len(keypoints) == 42:
            mp_pred = model_mp.predict(np.expand_dims(keypoints, axis=0), verbose=0)[0]
        else:
            mp_pred = np.zeros_like(cnn_pred)
        row = np.concatenate([cnn_pred, mp_pred]).tolist() + [class_name]
        features.append(row)
    print(f"类别 {class_name} 特征提取完成。")

with open(output_csv, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(features)

print(f"全部训练集特征已保存到 {output_csv}。")
