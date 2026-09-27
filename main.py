"""
main.py  -  CSE 535 SmartHome Gesture Control, Project Part 2
=============================================================
Recognises which of 17 hand gestures each test video shows, and writes the
answers to Results.csv (one number per line, no header). The autograder runs
only this file.

How it works
------------
1. Every video is turned into ONE feature vector:
      read 3 frames (at 25%, 50%, 75% of the video)
      -> run each frame through the CNN (handshape_feature_extractor.py)
      -> average the 3 outputs.
2. Do that for every training video (traindata/) and every test video (test/).
3. Subtract the average training vector from all vectors ("centring"), so the
   comparison focuses on what differs between gestures.
4. For each test video, find the training video whose vector is closest by
   cosine distance. That training video's gesture label is the prediction.
5. Write the predictions to Results.csv in sorted test-filename order.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import csv                                    # writes Results.csv
import os                                     # lists folders, splits file names

import cv2                                    # OpenCV: opens videos and reads frames
import numpy as np                            # averages / subtracts feature vectors
from scipy.spatial.distance import cosine     # cosine distance: 0 = same direction, 2 = opposite

from handshape_feature_extractor import HandShapeFeatureExtractor   # loads and runs the CNN

# import tensorflow as tf                     # NOT USED: keras is imported inside the extractor
# from keras.models import load_model         # NOT USED: the extractor loads the model

# model = load_model(model_path)              # NOT USED: loaded the model a second time;
# print("Model loaded successfully!")         # the extractor below already loads it once


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TRAIN_DIR = "traindata"          # labelled training videos (the file name holds the label)
TEST_DIR = "test"                # videos to classify (filled in by the autograder)
RESULTS_FILE = "Results.csv"     # output: one predicted label per test video

# Where in each video to grab frames (fraction of the video's length).
# Averaging 3 frames is more reliable than using only the middle frame.
FRAME_POSITIONS = (0.25, 0.5, 0.75)

# Gesture name (as it appears at the end of a file name) -> numeric label
# required by the assignment.
GESTURE_MAPPING = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "DecreaseFanSpeed": 10, "FanOff": 11, "FanOn": 12, "IncreaseFanSpeed": 13,
    "LightOff": 14, "LightOn": 15, "SetThermo": 16,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def gesture_label_from_filename(video_file):
    """Read the gesture label from a file name, e.g. "T1-H-FanOn.mp4" -> 12.

    Returns -1 if the name isn't recognised. Only understands the "...-<Gesture>.mp4"
    naming used by the course videos, not the Part 1 app's
    "FanDown_PRACTICE_1_Name.mp4" naming.
    """
    name = os.path.splitext(video_file)[0]        # drop the extension: "T1-H-FanOn"
    gesture = name.split("-")[-1]                  # keep the part after the last "-": "FanOn"
    if "Decerease" in gesture:                     # one course file name has this typo
        gesture = "DecreaseFanSpeed"
    return GESTURE_MAPPING.get(gesture, -1)        # look up the number; -1 if unknown


def video_feature(video_path, extractor):
    """Turn one video into one feature vector (the average of 3 frames' CNN outputs).

    Returns None if no frame could be read or processed.
    """
    # if not os.path.exists(video_path):          # NOT NEEDED: paths come from os.listdir,
    #     return None                             # so they always exist

    cap = cv2.VideoCapture(video_path)                         # open the video file
    if not cap.isOpened():                                     # not a readable video
        return None                                            # (e.g. a stray .gitkeep file)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))       # total number of frames

    frame_features = []                                        # one CNN output per frame
    for position in FRAME_POSITIONS:
        frame_index = min(frame_count - 1, int(frame_count * position))  # e.g. 25% -> frame 30 of 120
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)          # jump to that frame
        ok, frame = cap.read()                                 # read it (BGR image)
        if ok and frame is not None:
            feature = extractor.extract_feature(frame)         # CNN output, shape (1, 27)
            if feature is not None:
                frame_features.append(feature.flatten())       # store as a flat 27-number vector
    cap.release()                                              # close the video file

    if not frame_features:                                     # every frame failed
        return None
    return np.mean(frame_features, axis=0)                     # average the 3 vectors into 1


def load_training_set(extractor):
    """Build (labels, vectors) from every video in traindata/, one entry per video."""
    labels, vectors = [], []

    # if not os.listdir(TRAIN_DIR):                            # NOT NEEDED: an empty folder
    #     print("No videos found in traindata/")               # already fails with the error below

    for video_file in sorted(os.listdir(TRAIN_DIR)):           # sorted = same order on every machine
        vector = video_feature(os.path.join(TRAIN_DIR, video_file), extractor)
        if vector is None:
            print(f"Skipping training file {video_file}: could not read frames.")
            continue

        label = gesture_label_from_filename(video_file)
        if label == -1:
            print(f"Skipping training file {video_file}: unrecognised gesture name.")
            continue

        labels.append(label)          # keep every video (3 takes of a gesture = 3 entries)
        vectors.append(vector)

    if not vectors:
        raise ValueError("No valid training videos found in traindata/.")
    return labels, vectors


def nearest_label(test_vector, train_labels, train_vectors):
    """Return the label of the training vector with the smallest cosine distance."""
    best_label = None
    best_distance = float("inf")                               # start with "infinitely far"
    for label, train_vector in zip(train_labels, train_vectors):
        distance = cosine(test_vector, train_vector)           # 0 = most similar
        if distance < best_distance:                           # closer than anything so far?
            best_distance = distance
            best_label = label
    return best_label


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------
def main():
    # Load the CNN once; every frame of every video goes through this object.
    extractor = HandShapeFeatureExtractor.get_instance()

    # Steps 1-2 (training side): one averaged vector per training video.
    train_labels, train_vectors = load_training_set(extractor)
    # print(f"Loaded {len(train_vectors)} training videos.")  # progress message only

    # Step 3: centre everything on the average training vector. Every frame
    # shares a common component (similar backgrounds, lighting); removing it
    # lets cosine distance compare the parts that differ between gestures.
    feature_mean = np.mean(train_vectors, axis=0)
    train_vectors = [vector - feature_mean for vector in train_vectors]

    # Test videos, sorted so Results.csv rows line up with the grader's answer order.
    test_videos = sorted(f for f in os.listdir(TEST_DIR) if f.endswith(".mp4"))
    if not test_videos:
        # print("No test videos found in test/.")             # progress message only
        return

    # Steps 2 and 4 (test side): vector per test video -> closest training label.
    predictions = []
    for test_video in test_videos:
        test_vector = video_feature(os.path.join(TEST_DIR, test_video), extractor)
        if test_vector is None:
            # Skipping a test video shifts every later row up by one, so rows no
            # longer line up with the grader's answers. Only happens for unreadable videos.
            print(f"Skipping test file {test_video}: could not read frames.")
            continue
        test_vector = test_vector - feature_mean                # same centring as training

        label = nearest_label(test_vector, train_labels, train_vectors)
        predictions.append(label)
        # print(f"Recognized {test_video} as gesture {label}")  # progress message only

    # Step 5: one label per line, no header (the format the autograder expects).
    with open(RESULTS_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        for label in predictions:
            writer.writerow([label])
    # print(f"Results saved to {RESULTS_FILE}.")               # progress message only


# Run main() only when this file is executed directly (python main.py),
# not when another file imports it.
if __name__ == "__main__":
    main()
