import cv2
import numpy as np
import os

import keras
load_model = keras.models.load_model
Model = keras.models.Model

"""
This is a Singleton class which bears the ML model in memory.
"""

BASE = os.path.dirname(os.path.abspath(__file__))

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
                # Load the model (Ensure the model file exists)
                model_path = os.path.join(BASE, 'gestures_trained_cnn_model.keras')
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"⚠️ Model file not found: {model_path}")

                real_model = load_model(model_path)
                self.model = real_model
                HandShapeFeatureExtractor.__single = self
                print("✅ Model loaded successfully!")

            except Exception as e:
                print(f"❌ Error loading model: {str(e)}")
                raise

        else:
            raise Exception("This class bears the model, so it is a Singleton.")

    # Private method to preprocess the image
    @staticmethod
    def __pre_process_input_image(crop):
        try:
            # Validate input
            if crop is None or crop.size == 0:
                print("⚠️ Warning: Received an empty frame for processing.")
                return None  # Skip processing

            # The model was trained on RGB; OpenCV decodes frames as BGR.
            if crop.ndim == 3:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            # Resize the image to 300x300
            img = cv2.resize(crop, (300, 300))

            # Normalize pixel values (scale to [0,1])
            img_arr = np.array(img) / 255.0

            # Ensure the image has 3 color channels
            if len(img_arr.shape) == 2:  # Grayscale image
                img_arr = np.stack((img_arr,) * 3, axis=-1)

            # Reshape the array to match model input (1, 300, 300, 3)
            img_arr = img_arr.reshape(1, 300, 300, 3)
            return img_arr

        except Exception as e:
            print(f"❌ Error in preprocessing image: {str(e)}")
            return None  # Skip invalid images

    def extract_feature(self, image):
        try:
            img_arr = self.__pre_process_input_image(image)

            if img_arr is None:
                print("⚠️ Feature extraction skipped: Invalid image input.")
                return None

            return self.model.predict(img_arr)

        except Exception as e:
            print(f"❌ Error extracting features: {str(e)}")
            return None
