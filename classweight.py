import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load your dataset
train_data_dir = "C:/Users/giris/OneDrive/Desktop/mini/dataset_cleaned"

train_datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

class_weights_dict = dict(enumerate(class_weights))
print("\n📊 Class Weights:\n", class_weights_dict)

