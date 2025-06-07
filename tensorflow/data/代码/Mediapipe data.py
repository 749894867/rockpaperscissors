import cv2
import mediapipe as mp
import os
import pandas as pd
from tqdm import tqdm
import shutil

data_dir = "data/train_augmented"
failed_dir = "data/failed_samples"
os.makedirs(failed_dir, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

dataset = []
failed = []

for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    failed_class_path = os.path.join(failed_dir, class_name)
    os.makedirs(failed_class_path, exist_ok=True)

    label = class_name

    for img_name in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
        img_path = os.path.join(class_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y])
            keypoints.append(label)
            dataset.append(keypoints)
        else:
            failed.append(img_path)
            shutil.copy(img_path, os.path.join(failed_class_path, img_name))

hands.close()


columns = [f"x{i}" if i % 2 == 0 else f"y{i//2}" for i in range(42)]
columns.append("label")
df = pd.DataFrame(dataset, columns=columns)
df.to_csv("hand_keypoints_dataset.csv", index=False)

# 保存失败图像路径
with open("failed_images.txt", "w") as f:
    for path in failed:
        f.write(path + "\n")

print(f" 关键点提取完成，成功样本: {len(df)}, 失败样本: {len(failed)}")
print("所有失败图像已复制到 data/failed_samples/")
