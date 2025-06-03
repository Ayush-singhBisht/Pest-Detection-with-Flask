import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input

# Optional: Avoid full GPU memory allocation (helps prevent crash on some systems)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"GPU memory growth error: {e}")

# Define the ResNet50V2-based model structure
def create_resnet_model():
    base_model = ResNet50V2(input_shape=(224, 224, 3),
                            include_top=False,
                            weights='imagenet')
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

# Load the trained model or load weights if full model load fails
try:
    print("Trying to load full model...")
    model = load_model("ResNet50_Transfer_Learning.keras", compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Full model load failed: {e}")
    print("Reconstructing model and loading weights...")
    model = create_resnet_model()
    model.load_weights("ResNet50_Transfer_Learning.keras")
    print("Weights loaded successfully!")

# Class names for your dataset
class_names = ['Ant', 'Centipede', 'Cockroach', 'Fly', 'Human', 'Lizard', 'Mosquito', 'Moths', 'Rat', 'Snake', 'Spider', 'Wasp']

# Start webcam
cap = cv2.VideoCapture(0)  # 0 = Default camera
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting webcam... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess the frame
    img = cv2.resize(frame, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Prediction
    try:
        prediction = model.predict(img, verbose=0)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)
        label = f"{class_names[class_index]}: {confidence * 100:.2f}%"
    except Exception as e:
        label = f"Prediction error: {e}"

    # Display prediction
    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('Pest Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
