import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("crop_classifier_model.keras")

# Load class names
with open("class_names.json", "r") as f:
    class_indices = json.load(f)
class_names = {v: k for k, v in class_indices.items()}

# Load and preprocess test image
img = cv2.imread("test_leaf.jpg")
img = cv2.resize(img, (128, 128))
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)
predicted_index = np.argmax(prediction)
confidence = np.max(prediction)

# Print result
print(f"✅ Predicted Class: {class_names[predicted_index]}")
print(f"🔍 Confidence: {confidence:.2f}")
