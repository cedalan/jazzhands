import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Basic model for recognition of different joints
MODEL_PATH = "hand_model/hand_landmarker.task"

# We compare fingertips to knuckle joints (MCP).
# Fingertip coords (from model)
FINGER_TIPS = [4, 8, 12, 16, 20] # Thumb, index, middle, ring, pinky

# Knuckle coords (from model)
FINGER_MCP = [1, 5, 9, 13, 17] # Thumb, index, middle, ring, pinky

def is_finger_extended() -> bool:
    """
    Checks if a finger is extended by comparing tip coords with mcp coords
    """
    return True

def recognize_left_hand_gesture():
    """
    Classifies hand gesture based on left hand finger extention pattern.
    This is to classify which chord to play.
    """
    return

def recognize_right_hand_fingers():
    """
    Classifies which fingers are extended by the right hand to classify which
    chord extentions to add.
    """
    return

def draw_landmarks():
    """
    Draws landmarks to visualize how model interprets hand gestures.
    """
    return

def draw_gesture_and_extensions():
    """
    Adds text which tells which chord gesture is classified as and which extensions
    which are recognized.
    """
    return