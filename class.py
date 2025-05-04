import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

class_labels = [
    "Corn___Common_Rust", "Corn___Gray_Leaf_Spot", "Corn___Healthy", "Corn___Northern_Leaf_Blight",
    "Potato___Early_Blight", "Potato___Healthy", "Potato___Late_Blight",
    "Rice___Brown_Spot", "Rice___Healthy", "Rice___Leaf_Blast", "Rice___Neck_Blast",
    "Wheat___Brown_Rust", "Wheat___Healthy", "Wheat___Yellow_Rust",
    "Sugarcane__Red_Rot", "Sugarcane__Healthy", "Sugarcane__Bacterial_Blight"
]

# Load model
model = tf.keras.models.load_model("model.h5")

# Image path
img_path = "C:/Users/giris/OneDrive/Desktop/mini/leaf1.jpeg.JPG"

# Check if the file exists
print("Checking image path:", img_path)
if not os.path.exists(img_path):
    print("❌ Image file not found. Please check the path.")
    exit()

# Load and preprocess image

img = image.load_img(img_path, target_size=(128, 128))

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])

# Output
print("✅ Prediction complete!")
print("Predicted class index:", predicted_class)
print("Predicted class label:", class_labels[predicted_class])
