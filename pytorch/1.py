import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

class HandGestureDataset(Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img_t = self.transform(img)

        kpt_path = img_path.rsplit('.', 1)[0] + '.npy'
        if os.path.exists(kpt_path):
            kpt = np.load(kpt_path).astype(np.float32)
            mask = 1.0
        else:
            kpt = np.zeros(42, dtype=np.float32)
            mask = 0.0

        kpt = (kpt - 0.5) * 2
        kpt = np.concatenate([kpt, [mask]])

        return img_t, torch.from_numpy(kpt).float(), label, img_path


def get_samples(data_dir, class_to_idx):
    samples = []
    for cls in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(class_dir): continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('jpg', 'jpeg', 'png')):
                img_path = os.path.join(class_dir, fname)
                samples.append((img_path, class_to_idx[cls]))
    return samples


def draw_hand_keypoints_pil(
        img_pil, keypoints,
        point_color=(255, 0, 0),
        line_color=(0, 255, 255),
        radius=6,
        thickness=6
):

    img = img_pil.copy()
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    pts = ((keypoints / 2 + 0.5) * np.array([img_w, img_h] * 21)).reshape(-1, 2)
    pts = pts.astype(int)

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]

    for i, j in connections:
        draw.line([tuple(pts[i]), tuple(pts[j])], fill=line_color, width=thickness)
    for x, y in pts:
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=point_color)

    return img




class SEBlock(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        y = self.avg_pool(x.view(b, c, 1)).view(b, c)
        y = self.fc(y).view(b, c)
        return x * y.expand_as(x)


class GestureNet(nn.Module):

    def __init__(self, num_classes=4, kpt_dim=43, dropout_img=0.5, dropout_kpt=0.5):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights='DEFAULT')
        self.img_features = mobilenet.features
        self.img_gap = nn.AdaptiveAvgPool2d(1)
        self.img_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_img),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_img),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.kpt_mlp = nn.Sequential(
            nn.Linear(kpt_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_kpt),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_kpt)
        )
        fusion_dim = 128

        self.se_block = SEBlock(channel=fusion_dim, reduction=8)

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, num_classes)
        )
    def forward(self, img, kpt):
        x_img = self.img_features(img)
        x_img = self.img_gap(x_img).squeeze(-1).squeeze(-1)
        x_img = self.img_fc(x_img)

        x_kpt = self.kpt_mlp(kpt)
        x_fused = torch.cat([x_img, x_kpt], dim=1)

        x_fused_attention = self.se_block(x_fused)

        output = self.fusion(x_fused_attention)

        return output


class EarlyStopping:

    def __init__(self, patience=10, min_delta=0, save_path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_acc, model):
        if self.best_score is None or val_acc > self.best_score + self.min_delta:
            print(f"Validation accuracy improved ({self.best_score or -1:.4f} --> {val_acc:.4f}). Saving model...")
            self.best_score = val_acc
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def evaluate(model, loader, device, criterion, desc=""):
    model.eval()
    all_preds, all_labels = [], []
    loss_sum, total = 0, 0
    with torch.no_grad():
        for img, kpt, label, _ in tqdm(loader, desc=desc, leave=False):
            img, kpt, label = img.to(device), kpt.to(device), label.to(device)
            logits = model(img, kpt)
            loss = criterion(logits, label)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            loss_sum += loss.item() * img.size(0)
            total += img.size(0)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = (all_preds == all_labels).mean()
    avg_loss = loss_sum / total
    return avg_loss, acc, all_preds, all_labels


def tile_images(img_paths, n_cols=5, gap=10, bg_color=(255, 255, 255), out_size=(224, 224)):
    if not img_paths: return None
    imgs = [Image.open(p).convert('RGB').resize(out_size, Image.Resampling.BILINEAR) for p in img_paths]
    w, h = out_size
    n = len(imgs)
    n_rows = (n + n_cols - 1) // n_cols
    canvas = Image.new('RGB', ((w + gap) * n_cols - gap, (h + gap) * n_rows - gap), color=bg_color)
    for idx, im in enumerate(imgs):
        row, col = divmod(idx, n_cols)
        canvas.paste(im, (col * (w + gap), row * (h + gap)))
    return canvas

if __name__ == "__main__":
    train_dir = 'data/train_augmented'
    test_dir = 'data/test'
    batch_size = 16
    val_ratio = 0.25
    epochs = 100
    lr = 1e-3
    num_workers = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        print(f"Error: Make sure the data directories '{train_dir}' and '{test_dir}' exist.")
        exit()

    classes = sorted(os.listdir(train_dir))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}
    print(f"Using device: {device}")
    print(f"Classes: {classes}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    all_samples = get_samples(train_dir, class_to_idx)
    train_samples, val_samples = train_test_split(
        all_samples, test_size=val_ratio, stratify=[x[1] for x in all_samples], random_state=42
    )
    train_dataset = HandGestureDataset(train_samples, transform)
    val_dataset = HandGestureDataset(val_samples, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_samples = get_samples(test_dir, class_to_idx)
    test_dataset = HandGestureDataset(test_samples, transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = GestureNet(num_classes=len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopper = EarlyStopping(patience=10, save_path='best_model_with_se.pth')

    train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist = [], [], [], []

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss, correct, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for img, kpt, label, _ in pbar:
            img, kpt, label = img.to(device), kpt.to(device), label.to(device)

            optimizer.zero_grad()
            logits = model(img, kpt)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == label).sum().item()
            total += img.size(0)
            epoch_loss += loss.item() * img.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")

        train_loss = epoch_loss / total
        train_acc = correct / total

        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, criterion, desc="Validation")

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        print(
            f"\nEpoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        scheduler.step(val_loss)
        early_stopper(val_acc, model)

        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(early_stopper.save_path, map_location=device))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_hist, label='Train Loss')
    plt.plot(val_loss_hist, label='Val Loss')
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_hist, label='Train Acc')
    plt.plot(val_acc_hist, label='Val Acc')
    plt.legend()
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig('train_val_curve_with_se.png')
    plt.show()

    _, _, val_preds, val_labels = evaluate(model, val_loader, device, criterion, desc="Final Validation Eval")
    print("\n[Final Validation Metrics]")
    print(classification_report(val_labels, val_preds, target_names=classes))

    os.makedirs('results', exist_ok=True)
    model.eval()
    test_preds, test_labels = [], []

    print("\nRunning inference on the test set...")
    for img, kpt, label, path in tqdm(test_loader, desc="Testing"):
        img, kpt = img.to(device), kpt.to(device)
        with torch.no_grad():
            logits = model(img, kpt)
        pred = logits.argmax(dim=1).item()

        test_preds.append(pred)
        test_labels.append(label.item())

        pred_str = idx_to_class[pred]
        gt_str = idx_to_class[label.item()]

        if pred != label.item():
            print(f"错误预测: 文件名: {os.path.basename(path[0])}, 预测类别: {pred_str}, 真实类别: {gt_str}")

    print("\n[Final Test Metrics]")
    print(classification_report(test_labels, test_preds, target_names=classes))

    print("Generating confusion matrix...")
    cm = confusion_matrix(test_labels, test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap="Blues", xticks_rotation=45, ax=ax)
    plt.title("Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig("test_confusion_matrix_with_se.png")
    plt.show()

    print("\n脚本执行完毕。")