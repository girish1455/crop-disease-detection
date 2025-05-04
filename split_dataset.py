import os
import shutil
import random

base_dir = 'dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
split_ratio = 0.8

# Get all class folders in the base dataset
class_folders = [f for f in os.listdir(base_dir)
                 if os.path.isdir(os.path.join(base_dir, f)) and f not in ['train', 'val']]

for cls in class_folders:
    print(f"Splitting class: {cls}")
    full_path = os.path.join(base_dir, cls)
    images = os.listdir(full_path)
    random.shuffle(images)
    split_idx = int(len(images) * split_ratio)

    train_images = images[:split_idx]
    val_images = images[split_idx:]

    train_cls_path = os.path.join(train_dir, cls)
    val_cls_path = os.path.join(val_dir, cls)
    os.makedirs(train_cls_path, exist_ok=True)
    os.makedirs(val_cls_path, exist_ok=True)

    for img in train_images:
        shutil.move(os.path.join(full_path, img), os.path.join(train_cls_path, img))
    for img in val_images:
        shutil.move(os.path.join(full_path, img), os.path.join(val_cls_path, img))

    # Remove empty original folder
    os.rmdir(full_path)

print("✅ Dataset successfully split into 'train' and 'val'")
