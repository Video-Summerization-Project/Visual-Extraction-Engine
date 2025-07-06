# graph/steps/extract_features.py

import os
import cv2
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from types_.state import GraphState

def extract_frame_features(state: GraphState) -> GraphState:
    """Extracts visual features from the frame image."""
    frame_path = state["frame_path"]

    try:
        img = cv2.imread(frame_path)
        if img is None:
            state["frame_features"] = {"error": "Failed to load frame"}
            state["next_step"] = "evaluate_importance"
            return state

        height, width, channels = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        brightness = np.mean(gray)
        dark_pixels = np.sum(gray < 30) / (height * width)
        color_variance = np.var(img.reshape(-1, 3), axis=0).sum()

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        has_faces = len(faces) > 0

        state["frame_features"] = {
            "dimensions": {"height": height, "width": width},
            "contrast": float(contrast),
            "brightness": float(brightness),
            "dark_ratio": float(dark_pixels),
            "color_variance": float(color_variance),
            "has_faces": has_faces,
            "face_count": len(faces),
        }

        # Convert to base64
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        state["frame_data"] = {
            "base64_image": img_str,
            "file_name": os.path.basename(frame_path)
        }

    except Exception as e:
        print(f"Error extracting frame features: {str(e)}")
        state["frame_features"] = {"error": f"Feature extraction failed: {str(e)}"}
        state["frame_data"] = {"file_name": os.path.basename(frame_path)}

    state["next_step"] = "evaluate_importance"
    return state
