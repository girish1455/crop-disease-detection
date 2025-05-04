#Crop Disease Detection Using Custom CNN
This project is a deep learning-based image classification model built using a custom Convolutional Neural Network (CNN) architecture to detect and classify diseases in crop leaves. It uses the "Five Crop Diseases Dataset" and is developed using Python and TensorFlow in Visual Studio Code (VS Code).

The goal is to help farmers and researchers quickly identify plant diseases using images, so they can take timely and informed actions.

 #Dataset
I used the Five Crop Diseases Dataset from Kaggle. It includes images of diseased and healthy leaves of the following crops:

Corn

Potato

Rice

Wheat

Sugarcane

Each crop has multiple classes for different diseases (and healthy leaves).

# Features
Custom CNN model for image classification

Trained on 17 categories (diseases and healthy leaf classes)

Achieves high training and decent validation accuracy

Predicts the disease from an image and shows confidence score

Includes information on symptoms, cure, and prevention

Easily integrable with a web app or mobile interface (e.g., Streamlit or Flask)

#Tech Stack
Python

TensorFlow / Keras

NumPy, Matplotlib

VS Code

(Optional) Streamlit for web interface

#Model Summary
The CNN model includes:

Convolutional layers

Max Pooling

Dropout for regularization

Fully connected dense layers

Final softmax activation for multi-class classification

The model is saved as .h5 or .keras, and class names are stored in class_names.json.
*Extra Tips
first install all requriment.txt
pip install -r requirements.txt

python verison 3.8.0

 
