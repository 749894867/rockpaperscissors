import os
import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

save_dir = "mlp_fusion_only"
os.makedirs(save_dir, exist_ok=True)

train_csv = r"stacking_train_features.csv"
model_save_path = os.path.join(save_dir, "mlp_fusion_trained_on_trainset.joblib")

df_train = pd.read_csv(train_csv)
X_train = df_train.drop(columns=['label']).values
y_train = df_train['label'].values

mlp = MLPClassifier(hidden_layer_sizes=(100,),
                    max_iter=50,
                    random_state=42)
mlp.fit(X_train, y_train)

y_pred_train = mlp.predict(X_train)
print("\n=== MLP融合器在训练集上的表现 ===")
print(f"准确率: {accuracy_score(y_train, y_pred_train):.4f}")
print(classification_report(y_train, y_pred_train, digits=4))

# 绘制并保存混淆矩阵图片
labels = sorted(list(set(y_train) | set(y_pred_train)))
cm = confusion_matrix(y_train, y_pred_train, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("MLP Fusion Confusion Matrix (Train)")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "train_confusion_matrix_mlp.png"))
plt.show()

joblib.dump(mlp, model_save_path)
print(f"\n已保存MLP融合器模型至: {model_save_path}")
print("\n=== MLP融合器训练集评估与保存完成 ===")

