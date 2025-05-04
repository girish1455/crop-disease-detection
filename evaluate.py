import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Set paths
model_path = "crop_classifier_model.h5"
dataset_dir = "C:/Users/giris/OneDrive/Desktop/mini/dataset_cleaned"
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Load the trained model
model = load_model(model_path)

# Create validation generator from the cleaned dataset using the same settings as training
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_generator = val_datagen.flow_from_directory(
    dataset_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    subset='validation'
)

# Predict
predictions = model.predict(val_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# Generate classification report and confusion matrix
report = classification_report(y_true, y_pred, target_names=class_labels)
cmatrix = confusion_matrix(y_true, y_pred)

# Print to console
print("\n✅ Classification Report:\n")
print(report)
print("\n📊 Confusion Matrix:\n")
print(cmatrix)

# Save results to file
with open(os.path.join(results_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write("✅ Classification Report\n\n")
    f.write(report)

with open(os.path.join(results_dir, "confusion_matrix.txt"), "w", encoding="utf-8") as f:
    f.write("📊 Confusion Matrix\n\n")
    f.write(np.array2string(cmatrix))
