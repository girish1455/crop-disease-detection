from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import json

# 1. Load the trained model
model = load_model("crop_classifier_model.h5")

# 2. Load class label mapping from training
with open("class_names.json", "r") as f:
    class_indices = json.load(f)

# 3. Reverse the mapping: {index: class_label}
index_to_class = {v: k for k, v in class_indices.items()}

# 4. Load and preprocess the test image
img = image.load_img(r"C:\Users\giris\OneDrive\Desktop\mini\test_leaf.jpg", target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# 5. Predict
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions[0])
confidence = np.max(predictions[0])

# 6. Print result
print("✅ Predicted Class:", index_to_class[predicted_index])
print("🔍 Confidence: {:.2f}".format(confidence))
