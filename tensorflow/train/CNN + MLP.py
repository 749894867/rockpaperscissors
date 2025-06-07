import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import models, layers, callbacks, optimizers


data_dir = "data/train_augmented"
weights_dir = "weights"
os.makedirs(weights_dir, exist_ok=True)


cnn_model = MobileNetV2(include_top=False, pooling='avg', input_shape=(224, 224, 3))
cnn_model.trainable = False


def extract_feature(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = preprocess_input(np.expand_dims(x, axis=0))
        features = cnn_model.predict(x, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"处理图像 {img_path} 出错: {e}")
        return None


X, y = [], []
label_to_idx = {}
idx = 0
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    label_to_idx[class_name] = idx
    for img_name in tqdm(os.listdir(class_path), desc=f"处理 {class_name}"):
        img_path = os.path.join(class_path, img_name)
        feat = extract_feature(img_path)
        if feat is not None:
            X.append(feat)
            y.append(idx)
    idx += 1

X = np.array(X)
y = np.array(y)


X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.BatchNormalization(),

    layers.Dense(256, kernel_regularizer=regularizers.l2(1e-5)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),

    layers.Dense(128, kernel_regularizer=regularizers.l2(1e-5)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),

    layers.Dense(64, kernel_regularizer=regularizers.l2(1e-5)),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    layers.Dense(len(label_to_idx), activation='softmax')
])

optimizer = optimizers.Adam(learning_rate=1e-4, amsgrad=True)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=1000,
    batch_size=32,
    callbacks=[early_stop],
    verbose=2
)

model.save(os.path.join(weights_dir, "cnn_mlp_model.h5"))
joblib.dump(label_to_idx, os.path.join(weights_dir, "cnn_mlp_labels.joblib"))


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(weights_dir, "training_curves.png"))
plt.show()


y_pred = model.predict(X_val)
y_pred_labels = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_val, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_to_idx.keys()))
plt.figure(figsize=(6, 6))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(weights_dir, "confusion_matrix.png"))
plt.show()

print(" 完成：模型已保存至 'weights/'，训练曲线与混淆矩阵已生成")
