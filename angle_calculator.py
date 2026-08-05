# Angle calculation utilities for pose analysis

import numpy as np
from typing import Tuple, Dict, Optional


# Utility class for calculating angles between body joints
class AngleCalculator:
    # Calculate the angle between three points in degrees
    @staticmethod
    def calculate_angle(point1: Tuple[float, float],
                       point2: Tuple[float, float],
                       point3: Tuple[float, float]) -> float:
        # Convert to numpy arrays
        p1 = np.array(point1)
        p2 = np.array(point2)
        p3 = np.array(point3)

        # Calculate vectors
        vector1 = p1 - p2
        vector2 = p3 - p2

        # Calculate angle using dot product
        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

        # Clamp to avoid numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Convert to degrees
        angle = np.arccos(cos_angle)
        return np.degrees(angle)

    # Extract key joint angles from pose landmarks
    @staticmethod
    def extract_pose_angles(landmarks, pose_type: str = "standard") -> Dict[str, float]:
        if not landmarks:
            return {}

        angles = {}
        points = [(lm.x, lm.y) for lm in landmarks.landmark]

        if len(points) < 25:  # MediaPipe provides 33 landmarks
            return {}

        # Left elbow: shoulder-elbow-wrist
        angles['left_elbow'] = AngleCalculator.calculate_angle(
            points[11], points[13], points[15]
        )

        # Right elbow: shoulder-elbow-wrist
        angles['right_elbow'] = AngleCalculator.calculate_angle(
            points[12], points[14], points[16]
        )

        # Left shoulder: hip-shoulder-wrist (for arms up pose)
        angles['left_shoulder'] = AngleCalculator.calculate_angle(
            points[23], points[11], points[15]
        )

        # Right shoulder: hip-shoulder-wrist (for arms up pose)
        angles['right_shoulder'] = AngleCalculator.calculate_angle(
            points[24], points[12], points[16]
        )

        # Left knee: hip-knee-ankle
        angles['left_knee'] = AngleCalculator.calculate_angle(
            points[23], points[25], points[27]
        )

        # Right knee: hip-knee-ankle
        angles['right_knee'] = AngleCalculator.calculate_angle(
            points[24], points[26], points[28]
        )

        # Torso angle using hip center, shoulder center, and nose
        hip_center = (
            (points[23][0] + points[24][0]) / 2,
            (points[23][1] + points[24][1]) / 2
        )
        shoulder_center = (
            (points[11][0] + points[12][0]) / 2,
            (points[11][1] + points[12][1]) / 2
        )
        angles['torso'] = AngleCalculator.calculate_angle(
            hip_center, shoulder_center, points[0]
        )

        if pose_type == "detailed":
            # Add more angles for detailed analysis

            # Left hip: shoulder-hip-knee
            angles['left_hip'] = AngleCalculator.calculate_angle(
                points[11], points[23], points[25]
            )

            # Right hip: shoulder-hip-knee
            angles['right_hip'] = AngleCalculator.calculate_angle(
                points[12], points[24], points[26]
            )

            # Left ankle: knee-ankle-foot
            angles['left_ankle'] = AngleCalculator.calculate_angle(
                points[25], points[27], points[31]
            )

            # Right ankle: knee-ankle-foot
            angles['right_ankle'] = AngleCalculator.calculate_angle(
                points[26], points[28], points[32]
            )

        return angles

    # Calculate the absolute difference between two angles
    @staticmethod
    def calculate_angle_difference(angle1: float, angle2: float) -> float:
        return abs(angle1 - angle2)

    # Check if an angle is within tolerance of a reference angle
    @staticmethod
    def is_angle_within_tolerance(angle: float,
                                  reference: float,
                                  tolerance: float) -> bool:
        diff = AngleCalculator.calculate_angle_difference(angle, reference)
        return diff <= tolerance

    # Calculate overall similarity score between user and reference poses (0-100%)
    @staticmethod
    def calculate_similarity_score(user_angles: Dict[str, float],
                                   reference_angles: Dict[str, float],
                                   tolerance: float = 30.0) -> float:
        if not user_angles or not reference_angles:
            return 0.0

        total_score = 0.0
        joint_count = 0

        for joint, ref_angle in reference_angles.items():
            if joint in user_angles:
                user_angle = user_angles[joint]
                diff = AngleCalculator.calculate_angle_difference(user_angle, ref_angle)

                # Calculate score for this joint (0-100%)
                # Perfect match (diff=0) = 100%, tolerance=perfect, beyond tolerance=0%
                if diff <= tolerance:
                    joint_score = 100.0 * (1.0 - diff / tolerance)
                else:
                    # Gradual decrease beyond tolerance
                    joint_score = max(0.0, 100.0 * (1.0 - diff / 180.0))

                total_score += joint_score
                joint_count += 1

        return total_score / joint_count if joint_count > 0 else 0.0
