import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Any, List

# Basic model for recognition of different joints
MODEL_PATH = "hand_model/hand_landmarker.task"

# We compare fingertips to knuckle joints (MCP).
# Fingertip coords (from model)
FINGER_TIPS = [4, 8, 12, 16, 20] # Thumb, index, middle, ring, pinky

# Knuckle coords (from model)
FINGER_MCP = [2, 5, 9, 13, 17] # Thumb, index, middle, ring, pinky

def is_finger_extended(landmarks, finger_idx: int, is_right_hand: bool) -> bool:
    """
    Checks if a finger is extended by comparing tip coords with mcp coords

    Needs to check if is right hand for thumb logic.

    Returns:
    True if finger extended, False if finger not extended
    """

    tip = landmarks[FINGER_TIPS[finger_idx]]
    mcp = landmarks[FINGER_MCP[finger_idx]]

    if finger_idx == 0:
        if is_right_hand:
            # Is rip of hand extended:
            return tip.x > mcp.x
        else:
            return tip.x < mcp.x
    else:
        return tip.y < mcp.y

def recognize_left_hand_gesture(hand_landmarks) -> str:
    """
    Classifies hand gesture based on left hand finger extention pattern.
    This is to classify which chord to play.

    Returns:
    Chord symbol corresponding to the gesture given
    """

    is_right_hand = False

    extended = [is_finger_extended(hand_landmarks, i, is_right_hand) for i in range(5)]

    # See docs/hand-gestures.md for an explanation of gestures
    if all(extended):
        return "vii"
    
    if all(extended[1:]) and not extended[0]:
        return "vi"
    
    if all(extended[0:3]) and not extended[3] and not extended[4]:
        return "V"
    
    if not extended[0] and all(extended[1:3]) and not extended[3] and not extended[4]:
        return "IV"
    
    if all(extended[0:2]) and not extended[2] and not extended[3] and not extended[4]:
        return "iii"
    
    if extended[1] and not extended[0] and not extended[2] and not extended[3] and not extended[4]:
        return "ii"
    
    if extended[0] and not extended[1] and not extended[2] and not extended[3] and not extended[4]:
        return "I"
    
    return "UNKNOWN"
    

def recognize_right_hand_fingers(hand_landmarks) -> List[List[Any]]:
    """
    Classifies which fingers are extended by the right hand to classify which
    chord extentions to add.

    Returns:
    List containing extentions to add. Is empty if none
    """

    extensions = [
        ["7th", False],
        ["9th", False],
        ["??", False],
        ["??", False],
        ["13th", False]
    ]

    is_right_hand = True

    extended = [is_finger_extended(hand_landmarks, i, is_right_hand) for i in range(5)] 

    if extended[0]:
        extensions[0][1] = True

    if extended[1]:
        extensions[1][1] = True

    if extended[4]:
        extensions[4][1] = True

    return extensions

def draw_landmarks(rgb_image, detection_result):
    """
    Draws landmarks to visualize how model interprets hand gestures.
    """
    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape

    if detection_result.hand_landmarks:
        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            # Get handedness (Left or Right)
            handedness = detection_result.handedness[idx][0].category_name
            
            # Convert landmark coords to pixel coords
            landmark_points = []
            for landmark in hand_landmarks:
                x_px = min(int(landmark.x * width), width - 1)
                y_px = min(int(landmark.y * height), height - 1)
                landmark_points.append((x_px, y_px))
            
            # Draw connections (use predefined HAND_CONNECTIONS or define manually)
            # MediaPipe HandLandmarker's connections (same as before)
            connections = [
                (0,1),(1,2),(2,3),(3,4),       # thumb
                (0,5),(5,6),(6,7),(7,8),       # index
                (0,9),(9,10),(10,11),(11,12),  # middle
                (0,13),(13,14),(14,15),(15,16),# ring
                (0,17),(17,18),(18,19),(19,20),# pinky
                (5,9),(9,13),(13,17)           # palm
            ]
            for start, end in connections:
                cv2.line(annotated_image, landmark_points[start], landmark_points[end], (0,255,0), 2)
            
            # Draw landmarks
            for pt in landmark_points:
                cv2.circle(annotated_image, pt, 4, (0,0,255), -1)

            wrist = landmark_points[0]
            cv2.putText(annotated_image, handedness, (wrist[0] - 30, wrist[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

    return annotated_image

def draw_gesture_and_extensions(rgb_image, detection_result):
    """
    Adds text which tells which chord gesture is classified as and which extensions
    which are recognized.
    """
    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape

    if detection_result.hand_landmarks:
        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            # Get handedness (Left or Right)
            handedness = detection_result.handedness[idx][0].category_name
            
            # Convert landmark coords to pixel coords
            landmark_points = []
            for landmark in hand_landmarks:
                x_px = min(int(landmark.x * width), width - 1)
                y_px = min(int(landmark.y * height), height - 1)
                landmark_points.append((x_px, y_px))

            if handedness == "Right":
                extensions = recognize_right_hand_fingers(hand_landmarks)
                add_extension_text(annotated_image, landmark_points, extensions)

            else:
                chord_symbol = recognize_left_hand_gesture(hand_landmarks)
                add_chord_text(annotated_image, landmark_points, chord_symbol)


    return annotated_image

def add_chord_text(annotated_image, landmark_points, chord_symbol):
    """
    Adds chord text to image
    """
    thumb = landmark_points[4]
    cv2.putText(annotated_image, chord_symbol, (thumb[0], thumb[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

def add_extension_text(annotated_image, landmark_points, extensions):
    """
    Adds extension symbols over extended fingers
    """
    finger_tip_points = [landmark_points[tip_idx] for tip_idx in FINGER_TIPS]
    for i, pt in enumerate(finger_tip_points):
        if extensions[i][1] == True:
            ptx = pt[0]
            pty = pt[1]
            cv2.putText(annotated_image, extensions[i][0], (ptx + 10, pty + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

result_list = []
def print_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    result_list.append(result)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=print_result
)

with vision.HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(1)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        rgb_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        landmarker.detect_async(rgb_image, timestamp_ms)

        if result_list:
            current_result = result_list.pop(0)
            annotated_frame = draw_landmarks(frame, current_result)
            annotated_frame = draw_gesture_and_extensions(annotated_frame, current_result)
        else:
            annotated_frame = frame

        cv2.imshow('Hand Gesture Recognition', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()