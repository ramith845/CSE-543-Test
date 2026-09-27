"""
handshape_feature_extractor.py
==============================
Wraps the pre-trained gesture CNN so the rest of the project can turn one video
frame into one feature vector with a single call:

    extractor = HandShapeFeatureExtractor.get_instance()
    vector = extractor.extract_feature(frame)      # -> numpy array, shape (1, 27)

The model is loaded only once (singleton pattern), because loading it is slow
and every frame of every video goes through the same model.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os                      # builds the path to the model file

import cv2                     # OpenCV: resizing and grayscale conversion of frames
import keras                   # loads the saved CNN (Keras 3, same as the autograder)
import numpy as np             # turns the frame into a float array the model accepts

load_model = keras.models.load_model   # short name for the Keras loader
# Model = keras.models.Model           # NOT USED: only needed to build sub-models


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
# Folder this file lives in. Using it (instead of the current working directory)
# means the model is found no matter where main.py is launched from.
BASE = os.path.dirname(os.path.abspath(__file__))

# The model file. It is the course's cnn_model.h5 re-saved in the .keras format
# with Keras 3.12.2 (same layers, same weights), because the autograder's
# Keras 3.12.2 cannot load the old Keras 2 .h5 file, and cannot load a .keras
# saved by a newer Keras.
MODEL_FILE = "cnn_model.keras"
# MODEL_FILE = "gestures_trained_cnn_model.keras"  # other course model: needs 300 x 300 RGB input instead

# The CNN was trained on 200 x 200 grayscale (1-channel) images.
IMAGE_SIZE = 200


class HandShapeFeatureExtractor:
    """Holds the CNN in memory and converts frames into feature vectors."""

    # The single shared instance. Stays None until get_instance() is first called.
    __single = None

    # -----------------------------------------------------------------------
    # Singleton access
    # -----------------------------------------------------------------------
    @staticmethod
    def get_instance():
        """Return the one extractor, creating it (and loading the model) the first time."""
        if HandShapeFeatureExtractor.__single is None:   # first call?
            HandShapeFeatureExtractor()                   # __init__ stores itself in __single
        return HandShapeFeatureExtractor.__single         # every call returns the same object

    def __init__(self):
        # Refuse to build a second copy. Always use get_instance() instead of
        # calling HandShapeFeatureExtractor() directly.
        if HandShapeFeatureExtractor.__single is not None:
            raise Exception("This class bears the model, so it is a Singleton.")

        model_path = os.path.join(BASE, MODEL_FILE)       # full path to the model file
        if not os.path.exists(model_path):                # fail early with a clear message
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = load_model(model_path)               # load the CNN into memory
        HandShapeFeatureExtractor.__single = self         # remember this as the only instance
        # print("Model loaded successfully!")             # progress message only

    # -----------------------------------------------------------------------
    # Frame preparation
    # -----------------------------------------------------------------------
    @staticmethod
    def __pre_process_input_image(frame):
        """Convert one OpenCV frame into the exact input shape the CNN expects."""

        # if frame is None or frame.size == 0:            # NOT NEEDED: main.py only passes
        #     return None                                 # frames that were read successfully

        img = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))  # shrink/stretch to 200 x 200 pixels

        # OpenCV reads video frames as BGR colour (3 channels), but the model was
        # trained on grayscale. Without this line the reshape below fails:
        # 200 x 200 x 3 values cannot fit a 200 x 200 x 1 shape.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_arr = np.array(img) / 255.0                    # pixel values 0-255  ->  0.0-1.0

        # Add a leading "batch" dimension and a trailing "channel" dimension: the
        # model predicts on batches of images, so one gray image must have shape
        # (1, 200, 200, 1).
        return img_arr.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

    # -----------------------------------------------------------------------
    # Feature extraction
    # -----------------------------------------------------------------------
    def extract_feature(self, image):
        """Return the model's output for one frame, or None if the frame can't be processed.

        The output is the model's final layer: 27 numbers (one per class it was
        trained on). main.py compares these 27-number vectors with cosine distance.
        """
        try:
            img_arr = self.__pre_process_input_image(image)   # frame -> (1, 200, 200, 1) array
            return self.model.predict(img_arr, verbose=0)     # run the CNN; verbose=0 hides the progress bar
        except Exception:
            # print(f"Error extracting features: {e}")        # progress message only
            return None                                       # main.py skips frames that return None
