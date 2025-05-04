import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# --- CONFIG ---
data_dir = r'C:\Users\giris\OneDrive\Desktop\mini\dataset_cleaned'
img_size = (128, 128)
batch_size = 32
epochs = 10
model_path = 'crop_classifier_model.h5'  # 👈 Back to H5 format
label_map_path = 'class_names.json'

# --- DATA GENERATION ---
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# --- CLASS LABELS ---
class_labels = list(train_data.class_indices.keys())
print("🔖 Class Labels:", class_labels)

# --- COMPUTE CLASS WEIGHTS ---
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights_array))
print("\n📊 Class Weights:\n", class_weights)

# --- MODEL ARCHITECTURE ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- CALLBACKS ---
checkpoint = ModelCheckpoint(
    filepath=model_path,
    monitor='val_accuracy',
    save_best_only=True,
    save_format='h5'  # 👈 Forces it to save in H5 format
)

# --- TRAINING ---
model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[checkpoint],
    class_weight=class_weights
)

# --- SAVE CLASS LABELS ---
with open(label_map_path, 'w') as f:
    json.dump(train_data.class_indices, f)

print(f"\n✅ Model saved to: {model_path}")
print(f"📝 Class label mapping saved to: {label_map_path}")
