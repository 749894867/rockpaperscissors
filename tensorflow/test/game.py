import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import joblib

#
model_cnn = load_model("weights/cnn_mlp_model.h5")
cnn_labels = joblib.load("weights/cnn_mlp_labels.joblib")
idx_to_label_cnn = {v: k for k, v in cnn_labels.items()}

model_mp = load_model("weights_mediapipe/mediapipe_mlp_model.h5")
mp_labels = joblib.load("weights_mediapipe/mediapipe_mlp_labels.joblib")
idx_to_label_mp = {v: k for k, v in mp_labels.items()}

cnn_extractor = MobileNetV2(include_top=False, pooling='avg', input_shape=(224, 224, 3))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

def extract_keypoints(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    hand_data = []
    if result.multi_hand_landmarks and result.multi_handedness:
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            keypoints = np.array([[p.x, p.y] for p in hand_landmarks.landmark]).flatten()
            label = result.multi_handedness[i].classification[0].label  # 'Left' or 'Right'
            hand_data.append((label, keypoints, hand_landmarks))
    return hand_data

def extract_cnn_feature(img):
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img.astype(np.float32))
    return cnn_extractor.predict(np.expand_dims(img, axis=0), verbose=0)[0]


def predict_from_keypoints(keypoints):
    if len(keypoints) == 42:
        pred = model_mp.predict(np.expand_dims(keypoints, axis=0), verbose=0)
        return idx_to_label_mp[np.argmax(pred)]
    return None


def judge(left, right):
    rules = {'rock': 'scissors', 'scissors': 'paper', 'paper': 'rock'}
    if left == right:
        return "Draw"
    elif rules.get(left) == right:
        return "Left Wins"
    elif rules.get(right) == left:
        return "Right Wins"
    else:
        return "Invalid"


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" 摄像头打开失败")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow("Rock Paper Scissors", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print(" 无法读取帧")
        break

    h, w, _ = frame.shape
    mid = w // 2
    draw_frame = frame.copy()


    hand_data = extract_keypoints(frame)

    left_label = "None"
    right_label = "None"

    for label, keypoints, landmarks in hand_data:
        cx = np.mean([p.x for p in landmarks.landmark]) * w
        side = "left" if cx < mid else "right"

        gesture = predict_from_keypoints(keypoints)
        if gesture is None:
            feat = extract_cnn_feature(frame)
            gesture = idx_to_label_cnn[np.argmax(model_cnn.predict(np.expand_dims(feat, axis=0), verbose=0))]

        if side == "left":
            left_label = gesture
        else:
            right_label = gesture


        mp_drawing.draw_landmarks(draw_frame, landmarks, mp_hands.HAND_CONNECTIONS)


    result = judge(left_label, right_label)


    cv2.putText(draw_frame, f"Left: {left_label}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 100, 100), 3)
    cv2.putText(draw_frame, f"Right: {right_label}", (mid + 30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 255), 3)
    cv2.putText(draw_frame, f"Result: {result}", (mid - 150, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50, 255, 50), 4)
    cv2.line(draw_frame, (mid, 0), (mid, h), (180, 180, 180), 3)
    cv2.putText(draw_frame, "Player 1", (mid // 4, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(draw_frame, "Player 2", (mid + mid // 4, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    cv2.imshow("Rock Paper Scissors", draw_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
