import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input
from PIL import Image

# GPU memory optimization
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"GPU memory growth error: {e}")

# Class labels
class_names = ['Ant', 'Centipede', 'Cockroach', 'Fly', 'Human', 'Lizard',
               'Mosquito', 'Moths', 'Rat', 'Snake', 'Spider', 'Wasp']

def create_resnet_model():
    base_model = ResNet50V2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model = Sequential([
        Input(shape=(224, 224, 3)),
        base_model,
        Dropout(0.25),
        BatchNormalization(),
        Flatten(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(12, activation='softmax')
    ])
    return model

# Load model
try:
    print("Trying to load full model...")
    model = load_model("backend/ResNet50_Transfer_Learning.keras", compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Full model load failed: {e}")
    print("Reconstructing model and loading weights...")
    model = create_resnet_model()
    model.load_weights("backend/ResNet50_Transfer_Learning.keras")
    print("Weights loaded successfully!")

def predict_uploaded_image(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        label = f"{class_names[class_index]}: {confidence * 100:.2f}%"
        return label

    except Exception as e:
        return f"Prediction error: {e}"
