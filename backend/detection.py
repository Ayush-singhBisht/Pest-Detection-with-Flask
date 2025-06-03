import cv2
import numpy as np
import tensorflow as tf
import pygame
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input
from backend.logger import log_detection  

# # Initialize pygame for sound
pygame.mixer.init()
warning_sound = pygame.mixer.Sound("backend/warning.wav")

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"GPU memory growth error: {e}")

# Class names and danger categories
class_names = ['Ant', 'Centipede', 'Cockroach', 'Fly', 'Human', 'Lizard',
               'Mosquito', 'Moths', 'Rat', 'Snake', 'Spider', 'Wasp']

red_warning = {'Cockroach', 'Fly', 'Lizard', 'Rat', 'Snake'}
yellow_warning = {'Mosquito', 'Moths', 'Spider'}

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

# # OpenCV camera stream
# camera = cv2.VideoCapture(2)

# def gen_frames():
#     sound_played = False
#     last_logged_class = None
#     last_logged_confidence = 0.0
#     confidence_threshold = 0.05

#     while True:
#         success, frame = camera.read()
#         if not success:
#             break

#         img = cv2.resize(frame, (224, 224))
#         img = img.astype('float32') / 255.0
#         img = np.expand_dims(img, axis=0)

#         try:
#             prediction = model.predict(img, verbose=0)
#             class_index = np.argmax(prediction)
#             confidence = float(np.max(prediction))
#             class_name = class_names[class_index]
#             label = f"{class_name}: {confidence * 100:.2f}%"


#             if (class_name != last_logged_class) or (abs(confidence - last_logged_confidence) > confidence_threshold):
#                 log_detection(class_name, confidence, source='Live')
#                 last_logged_class = class_name
#                 last_logged_confidence = confidence

#         except Exception as e:
#             label = f"Prediction error: {e}"
#             class_name = ""
#             confidence = 0.0


#         if class_name in red_warning:
#             color = (0, 0, 255)  # Red
#             if not sound_played:
#                 try:
#                     warning_sound.play()
#                     sound_played = True
#                 except Exception as e:
#                     print(f"Sound error: {e}")
#         elif class_name in yellow_warning:
#             color = (0, 255, 255)  # Yellow
#             sound_played = False
#         else:
#             color = (0, 255, 0)  # Green
#             sound_played = False


#         cv2.putText(frame, label, (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')














# Camera initialization
camera1 = cv2.VideoCapture(0)
camera2 = cv2.VideoCapture(1)
camera3 = cv2.VideoCapture(2)

if not camera1.isOpened():
    print("Camera 1 not available")
if not camera2.isOpened():
    print("Camera 2 not available")
if not camera3.isOpened():
    print("Camera 3 not available")

def gen_frames(camera, cam_id="Camera"):
    sound_played = False
    last_logged_class = None
    last_logged_confidence = 0.0
    confidence_threshold = 0.05

    while True:
        success, frame = camera.read()
        if not success:
            break

        img = cv2.resize(frame, (224, 224))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        try:
            prediction = model.predict(img, verbose=0)
            class_index = np.argmax(prediction)
            confidence = float(np.max(prediction))
            class_name = class_names[class_index]
            label = f"{class_name}: {confidence * 100:.2f}%"

            # if (class_name != last_logged_class) or (abs(confidence - last_logged_confidence) > confidence_threshold):
            #     log_detection(class_name, confidence, source=cam_id)
            #     last_logged_class = class_name
            #     last_logged_confidence = confidence
            
            if class_name.lower() != "human" and (
                    (class_name != last_logged_class) or (abs(confidence - last_logged_confidence) > confidence_threshold)
                ):
                    log_detection(class_name, confidence, source=cam_id)
                    last_logged_class = class_name
                    last_logged_confidence = confidence


        except Exception as e:
            label = f"Prediction error: {e}"
            class_name = ""
            confidence = 0.0

        if class_name in red_warning:
            color = (0, 0, 255)
            if not sound_played:
                try:
                    warning_sound.play()
                    sound_played = True
                except Exception as e:
                    print(f"Sound error: {e}")
        elif class_name in yellow_warning:
            color = (0, 255, 255)
            sound_played = False
        else:
            color = (0, 255, 0)
            sound_played = False

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
