import cv2
import numpy as np
import csv
import os
import tensorflow as tf
from keras.models import load_model
from scipy.spatial.distance import cosine

# Load the model
model_path = "gestures_trained_cnn_model.keras"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Error: Model file '{model_path}' not found!")

model = load_model(model_path)
print("✅ Model loaded successfully!")

# Predefined mapping for gesture labels
gesture_mapping = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "DecreaseFanSpeed": 10, "FanOff": 11, "FanOn": 12, "IncreaseFanSpeed": 13,
    "LightOff": 14, "LightOn": 15, "SetThermo": 16
}

# Dictionary to store training features
training_features = {}

class HandShapeFeatureExtractor:
    __single = None

    @staticmethod
    def get_instance():
        if HandShapeFeatureExtractor.__single is None:
            HandShapeFeatureExtractor()
        return HandShapeFeatureExtractor.__single

    def __init__(self):
        if HandShapeFeatureExtractor.__single is None:
            try:
                self.model = load_model(model_path)
                HandShapeFeatureExtractor.__single = self
                print("✅ Model loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading model: {str(e)}")
                raise

    def extract_feature(self, image):
        try:
            resized_frame = cv2.resize(image, (300, 300))
            normalized_frame = resized_frame / 255.0
            input_frame = np.expand_dims(normalized_frame, axis=0)
            return self.model.predict(input_frame).flatten()
        except Exception as e:
            print(f"❌ Error extracting features: {str(e)}")
            return None

# Function to extract features using CNN model
def extract_features(frame):
    extractor = HandShapeFeatureExtractor.get_instance()
    return extractor.extract_feature(frame)

# Function to process videos and extract middle frame
def process_video(video_path):
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file '{video_path}' not found!")
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return None
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_idx = frame_count // 2  
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        print(f"⚠️ Skipping {video_path}: Cannot read middle frame.")
        return None
    
    return extract_features(frame)

# Load training video features from 'traindata/'
def load_training_features(folder_path):
    if not os.path.exists(folder_path) or not os.listdir(folder_path):
        print(f"⚠️ Warning: No videos found in '{folder_path}/' folder!")
        return

    print(f"✅ Loading gesture features from {folder_path}...")
    for video_file in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video_file)
        features = process_video(video_path)

        if features is not None:
            file_name = os.path.splitext(video_file)[0]  
            parts = file_name.split("-")  
            gesture_name = parts[-1]  

            if "Decerease" in gesture_name:
                gesture_name = "DecreaseFanSpeed"

            label = gesture_mapping.get(gesture_name, -1)  
            if label == -1:
                print(f"⚠️ Warning: Unrecognized gesture {video_file}, skipping...")
                continue

            training_features[label] = features

# Load training gestures
load_training_features("traindata")

if not training_features:
    raise ValueError("❌ Error: No valid training features extracted!")

print(f"✅ Loaded {len(training_features)} training gestures.")

# Process test videos
test_data_path = "test"
if not os.path.exists(test_data_path) or not os.listdir(test_data_path):
    print("⚠️ No test videos found in 'test/' folder! Skipping recognition.")
    exit()

print("✅ Processing test videos...")
recognized_gestures = []
test_videos = [f for f in os.listdir(test_data_path) if f.endswith(".mp4")]

for test_video in test_videos:
    test_video_path = os.path.join(test_data_path, test_video)
    test_features = process_video(test_video_path)
    if test_features is None:
        continue
    
    best_match = None
    best_score = float("inf")
    
    for train_label, train_features in training_features.items():
        similarity_score = cosine(test_features, train_features)  

        # 🔍 Print similarity for debugging
        print(f"🔍 Comparing with gesture {train_label} -> Cosine Similarity: {similarity_score:.5f}")

        if similarity_score < best_score:
            best_score = similarity_score
            best_match = train_label
    
    recognized_gestures.append(best_match)
    print(f"✅ Recognized {test_video} as gesture {best_match} (Score: {best_score:.5f})")

# Save results in required format (51x1 matrix, no headers)
output_file = "Results.csv"
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    for recognized_gesture in recognized_gestures:
        writer.writerow([recognized_gesture])  

print(f"🎉 Gesture recognition complete. Results saved to {output_file}.")
