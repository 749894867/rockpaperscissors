import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import models, layers, callbacks, optimizers


data_path = "hand_keypoints_dataset.csv"
weights_dir = "weights_mediapipe"
os.makedirs(weights_dir, exist_ok=True)

df = pd.read_csv(data_path)
X = df.drop(columns=["label"]).values
y_labels = df["label"].values

label_set = sorted(list(set(y_labels)))
label_to_idx = {label: idx for idx, label in enumerate(label_set)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
y = np.array([label_to_idx[label] for label in y_labels])

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(label_to_idx), activation='softmax')
])

optimizer = optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    min_delta=0.001,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=2
)

model.save(os.path.join(weights_dir, "mediapipe_mlp_model.h5"))
joblib.dump(label_to_idx, os.path.join(weights_dir, "mediapipe_mlp_labels.joblib"))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(weights_dir, "mediapipe_training_curves.png"))
plt.show()

y_pred = model.predict(X_val)
y_pred_labels = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_val, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_set)
plt.figure(figsize=(6, 6))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(weights_dir, "mediapipe_confusion_matrix.png"))
plt.show()

print("Mediapipe + MLP 训练完成，模型和评估图已保存到 weights/")
