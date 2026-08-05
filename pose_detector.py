# Pose detection module using MediaPipe

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple


# Handles pose detection using MediaPipe
class PoseDetector:
    # Initialize the pose detector with MediaPipe configuration
    def __init__(self,
                 static_mode: bool = False,
                 model_complexity: int = 0,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_mode,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.last_landmarks = None

    # Detect pose in the given image and return MediaPipe results
    def detect_pose(self, image: np.ndarray, convert_color: bool = True) -> Optional[object]:
        if convert_color:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image

        results = self.pose.process(rgb_image)

        if results.pose_landmarks:
            self.last_landmarks = results.pose_landmarks

        return results

    # Draw pose landmarks on the image
    def draw_landmarks(self,
                       image: np.ndarray,
                       landmarks,
                       connections: bool = True) -> np.ndarray:
        if landmarks is None:
            return image

        annotated_image = image.copy()

        if connections:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        else:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                landmarks,
                None,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        return annotated_image

    # Extract landmark points as (x, y) coordinates
    def get_landmark_points(self, landmarks) -> list:
        if landmarks is None:
            return []

        return [(lm.x, lm.y) for lm in landmarks.landmark]

    # Get specific landmark by index
    def get_landmark_by_index(self, landmarks, index: int) -> Optional[Tuple[float, float]]:
        if landmarks is None or index >= len(landmarks.landmark):
            return None

        lm = landmarks.landmark[index]
        return (lm.x, lm.y)

    # Release resources
    def cleanup(self):
        self.pose.close()
