import os
import cv2
import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import mediapipe as mp
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

test_dir = "data/test"
cnn_model_path = "weights/cnn_mlp_model.h5"
cnn_label_path = "weights/cnn_mlp_labels.joblib"
mp_model_path = "weights_mediapipe/mediapipe_mlp_model.h5"
mlp_fusion_model_path = "mlp_fusion_only/mlp_fusion_trained_on_trainset.joblib"
save_dir = "mlp_fusion_only"
os.makedirs(save_dir, exist_ok=True)

cnn_feat_model = MobileNetV2(include_top=False, pooling='avg', input_shape=(224, 224, 3))
cnn_feat_model.trainable = False
model_cnn = load_model(cnn_model_path)
cnn_labels = joblib.load(cnn_label_path)
idx_to_label_cnn = {v: k for k, v in cnn_labels.items()}
model_mp = load_model(mp_model_path)

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

X_test = []
y_true = []
img_names = []
total_imgs = 0

for class_name in sorted(os.listdir(test_dir)):
    class_path = os.path.join(test_dir, class_name)
    if not os.path.isdir(class_path): continue
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        cnn_feat = extract_cnn_feature(img)
        cnn_prob = model_cnn.predict(np.expand_dims(cnn_feat, axis=0), verbose=0)[0]
        keypoints = extract_keypoints(img)
        if keypoints is not None and len(keypoints) == 42:
            mp_prob = model_mp.predict(np.expand_dims(keypoints, axis=0), verbose=0)[0]
        else:
            mp_prob = np.zeros_like(cnn_prob)
        X_test.append(np.concatenate([cnn_prob, mp_prob]))
        y_true.append(class_name)
        img_names.append(img_name)
        total_imgs += 1

mlp_model = joblib.load(mlp_fusion_model_path)
X_test = np.array(X_test)

y_pred = []
print("\n预测结果：")
for i in range(len(X_test)):
    pred_label = mlp_model.predict(X_test[i].reshape(1, -1))[0]
    y_pred.append(pred_label)
    print(f"图片 {i+1}/{len(X_test)} - 文件名: {img_names[i]}, 真实类别: {y_true[i]}, 预测类别: {pred_label}")

print(f"准确率: {accuracy_score(y_true, y_pred):.4f}")
print(classification_report(y_true, y_pred, digits=4))

labels = sorted(list(set(y_true) | set(y_pred)))
cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("MLP Fusion Confusion Matrix (Test)")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "test_confusion_matrix_mlp.png"))
plt.show()

df_pred = pd.DataFrame({
    "file_name": img_names,
    "true_label": y_true,
    "pred_label": y_pred
})
df_pred.to_csv(os.path.join(save_dir, "mlp_test_predictions.csv"), index=False)
print(f"预测结果保存至: {os.path.join(save_dir, 'mlp_test_predictions.csv')}")
print(f"混淆矩阵图片保存至: {os.path.join(save_dir, 'test_confusion_matrix_mlp.png')}")

