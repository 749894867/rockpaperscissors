import os
import cv2
import albumentations as A
from tqdm import tqdm

input_dir = "data/train"
output_dir = "data/train_augmented"
augment_per_image = 9

transform = A.Compose([
    A.Resize(224, 224),
    A.OneOf([
        A.Rotate(limit=30),
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15),
    ], p=0.8),
    A.OneOf([
        A.GaussianBlur(blur_limit=3),
        A.MotionBlur(blur_limit=3),
        A.GaussNoise(),
        A.MedianBlur(blur_limit=3),
    ], p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(),
        A.CLAHE(),
        A.HueSaturationValue(),
        A.ColorJitter(),
    ], p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomShadow(p=0.3),
    A.RandomFog(p=0.3),
    A.Normalize(mean=(0.5,), std=(0.5,))
])

os.makedirs(output_dir, exist_ok=True)
for class_name in os.listdir(input_dir):
    class_input_path = os.path.join(input_dir, class_name)
    class_output_path = os.path.join(output_dir, class_name)
    os.makedirs(class_output_path, exist_ok=True)

    for img_name in tqdm(os.listdir(class_input_path), desc=f"Processing {class_name}"):
        img_path = os.path.join(class_input_path, img_name)
        base_name = os.path.splitext(img_name)[0]
        try:
            image = cv2.imread(img_path)
            if image is None:
                print(f"[警告] 无法读取图像: {img_path}")
                continue

            orig_resized = cv2.resize(image, (224, 224))
            save_path = os.path.join(class_output_path, f"{base_name}_orig.jpg")
            cv2.imwrite(save_path, orig_resized)

            for i in range(augment_per_image):
                augmented = transform(image=image)["image"]
                save_name = f"{base_name}_aug{i}.jpg"
                save_path = os.path.join(class_output_path, save_name)
                aug_img = ((augmented + 0.5) * 255).clip(0, 255).astype('uint8')
                cv2.imwrite(save_path, aug_img)

        except Exception as e:
            print(f" 图像增强失败: {img_path}，错误信息: {e}")
