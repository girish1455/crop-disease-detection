import os
import shutil

def move_images(src_root, dest_root):
    for root, dirs, files in os.walk(src_root):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                class_name = os.path.basename(root)
                dest_dir = os.path.join(dest_root, class_name)
                os.makedirs(dest_dir, exist_ok=True)
                src_path = os.path.join(root, file)
                dest_path = os.path.join(dest_dir, file)
                shutil.copy2(src_path, dest_path)

src_dirs = [
    r"C:\Users\giris\OneDrive\Desktop\mini\dataset\train",
    r"C:\Users\giris\OneDrive\Desktop\mini\dataset\val"
]
dest_dir = r"C:\Users\giris\OneDrive\Desktop\mini\dataset_cleaned"

# Clean the destination folder first (optional)
if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)
os.makedirs(dest_dir, exist_ok=True)

# Move all images from train and val folders
for src in src_dirs:
    move_images(src, dest_dir)

print(f"✅ Images copied successfully to: {dest_dir}")
