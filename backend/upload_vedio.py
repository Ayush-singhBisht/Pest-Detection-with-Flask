import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input
import time
import os

# Enable memory growth for GPU (optional)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"GPU memory growth error: {e}")

# Class names (in order of model's output)
class_names = ['Ant', 'Centipede', 'Cockroach', 'Fly', 'Human', 'Lizard',
               'Mosquito', 'Moths', 'Rat', 'Snake', 'Spider', 'Wasp']

# Build model structure (used if full model fails to load)
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
        Dense(len(class_names), activation='softmax')
    ])
    return model

# Load the model
model_path = "backend/ResNet50_Transfer_Learning.keras"
try:
    print("Trying to load full model...")
    model = load_model(model_path, compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Full model load failed: {e}")
    print("Reconstructing model and loading weights...")
    model = create_resnet_model()
    model.load_weights(model_path)
    print("Weights loaded successfully!")

# Generator to stream video frames with predictions
def gen_video_frames(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path not found: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception(f"Cannot open video file: {video_path}")

    while True:
        success, frame = cap.read()

        # Loop video if it ends
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        try:
            # Preprocess frame
            img = cv2.resize(frame, (224, 224))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)

            # Prediction
            prediction = model.predict(img, verbose=0)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

            # Format label
            if confidence > 0.5:
                label = f"{class_names[class_index]}: {confidence * 100:.2f}%"
            else:
                label = "Uncertain"

        except Exception as e:
            label = f"Prediction error: {e}"

        # Overlay label
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield MJPEG frame
        try:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except GeneratorExit:
            break

        # Control frame rate
        time.sleep(1 / 30)

    cap.release()
