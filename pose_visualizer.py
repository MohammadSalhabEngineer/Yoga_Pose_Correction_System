# Advanced pose visualization module for heatmaps and confidence scores

import cv2
import numpy as np
from typing import Dict, Tuple, Optional


# Advanced visualization tools for pose analysis
class PoseVisualizer:
    # Color scheme for heatmap (BGR format for OpenCV)
    COLORS = {
        'excellent': (0, 255, 0),      # Green - >90%
        'good': (0, 255, 255),         # Yellow - 70-90%
        'fair': (0, 165, 255),         # Orange - 50-70%
        'poor': (0, 0, 255),           # Red - <50%
        'neutral': (200, 200, 200)     # Gray - no data
    }

    # Get color based on score percentage
    @staticmethod
    def get_color_for_score(score: float) -> Tuple[int, int, int]:
        if score >= 90.0:
            return PoseVisualizer.COLORS['excellent']
        elif score >= 70.0:
            return PoseVisualizer.COLORS['good']
        elif score >= 50.0:
            return PoseVisualizer.COLORS['fair']
        else:
            return PoseVisualizer.COLORS['poor']

    # Draw skeleton with color-coded heatmap based on joint accuracy
    @staticmethod
    def draw_heatmap_skeleton(frame: np.ndarray,
                             landmarks,
                             joint_scores: Dict[str, dict],
                             mp_pose,
                             mp_drawing) -> np.ndarray:
        if not landmarks:
            return frame

        # Get image dimensions
        height, width = frame.shape[:2]

        # Map joint names to landmark indices
        joint_landmark_map = {
            'left_elbow': [mp_pose.PoseLandmark.LEFT_SHOULDER,
                          mp_pose.PoseLandmark.LEFT_ELBOW,
                          mp_pose.PoseLandmark.LEFT_WRIST],
            'right_elbow': [mp_pose.PoseLandmark.RIGHT_SHOULDER,
                           mp_pose.PoseLandmark.RIGHT_ELBOW,
                           mp_pose.PoseLandmark.RIGHT_WRIST],
            'left_knee': [mp_pose.PoseLandmark.LEFT_HIP,
                         mp_pose.PoseLandmark.LEFT_KNEE,
                         mp_pose.PoseLandmark.LEFT_ANKLE],
            'right_knee': [mp_pose.PoseLandmark.RIGHT_HIP,
                          mp_pose.PoseLandmark.RIGHT_KNEE,
                          mp_pose.PoseLandmark.RIGHT_ANKLE],
        }

        # Draw connections with color coding
        connections = [
            # Arms
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, 'left_elbow'),
            (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST, 'left_elbow'),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, 'right_elbow'),
            (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST, 'right_elbow'),
            # Legs
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, 'left_knee'),
            (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, 'left_knee'),
            (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, 'right_knee'),
            (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE, 'right_knee'),
            # Torso
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, 'torso'),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, 'torso'),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, 'torso'),
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP, 'torso'),
        ]

        # Draw colored connections
        for start_idx, end_idx, joint_name in connections:
            start = landmarks.landmark[start_idx]
            end = landmarks.landmark[end_idx]

            # Get color based on joint score
            if joint_name in joint_scores:
                score = joint_scores[joint_name]['score']
                color = PoseVisualizer.get_color_for_score(score)
            else:
                color = PoseVisualizer.COLORS['neutral']

            # Convert normalized coordinates to pixel coordinates
            start_pos = (int(start.x * width), int(start.y * height))
            end_pos = (int(end.x * width), int(end.y * height))

            # Draw line with thickness based on score (thicker = better)
            thickness = 4 if joint_name in joint_scores and joint_scores[joint_name]['score'] >= 70 else 3
            cv2.line(frame, start_pos, end_pos, color, thickness)

        # Draw joints as circles
        for joint_name, score_data in joint_scores.items():
            if joint_name in joint_landmark_map:
                # Get the middle landmark (the actual joint)
                landmark_idx = joint_landmark_map[joint_name][1]
                landmark = landmarks.landmark[landmark_idx]

                pos = (int(landmark.x * width), int(landmark.y * height))
                score = score_data['score']
                color = PoseVisualizer.get_color_for_score(score)

                # Draw joint circle
                cv2.circle(frame, pos, 8, color, -1)
                cv2.circle(frame, pos, 9, (255, 255, 255), 1)

        return frame

    # Draw color legend for heatmap
    @staticmethod
    def draw_heatmap_legend(frame: np.ndarray, position: str = "top-right") -> np.ndarray:
        height, width = frame.shape[:2]

        # Legend settings
        legend_width = 200
        legend_height = 140
        padding = 10

        # Calculate position
        if position == "top-right":
            x = width - legend_width - padding
            y = padding
        elif position == "top-left":
            x = padding
            y = padding
        elif position == "bottom-right":
            x = width - legend_width - padding
            y = height - legend_height - padding
        else:  # bottom-left
            x = padding
            y = height - legend_height - padding

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + legend_width, y + legend_height),
                     (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw border
        cv2.rectangle(frame, (x, y), (x + legend_width, y + legend_height),
                     (255, 255, 255), 2)

        # Draw title
        cv2.putText(frame, "Pose Quality", (x + 10, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw color indicators
        indicators = [
            (PoseVisualizer.COLORS['excellent'], "Excellent (>90%)"),
            (PoseVisualizer.COLORS['good'], "Good (70-90%)"),
            (PoseVisualizer.COLORS['fair'], "Fair (50-70%)"),
            (PoseVisualizer.COLORS['poor'], "Poor (<50%)")
        ]

        y_offset = y + 45
        for color, label in indicators:
            # Draw color box
            cv2.rectangle(frame, (x + 10, y_offset), (x + 30, y_offset + 15),
                         color, -1)
            cv2.rectangle(frame, (x + 10, y_offset), (x + 30, y_offset + 15),
                         (255, 255, 255), 1)

            # Draw label
            cv2.putText(frame, label, (x + 40, y_offset + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            y_offset += 25

        return frame

    # Draw confidence scores for pose detection
    @staticmethod
    def draw_confidence_scores(frame: np.ndarray,
                               landmarks,
                               overall_confidence: float = None) -> np.ndarray:
        height, width = frame.shape[:2]

        # Draw overall confidence bar at top center
        if overall_confidence is not None or landmarks:
            # Calculate average confidence from visible landmarks
            if landmarks:
                visible_confidences = [lm.visibility for lm in landmarks.landmark if lm.visibility > 0]
                avg_confidence = sum(visible_confidences) / len(visible_confidences) if visible_confidences else 0
            else:
                avg_confidence = overall_confidence or 0

            # Draw confidence bar
            bar_width = 300
            bar_height = 30
            bar_x = (width - bar_width) // 2
            bar_y = 10

            # Background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (50, 50, 50), -1)

            # Confidence fill
            fill_width = int(bar_width * avg_confidence)
            conf_color = (0, 255, 0) if avg_confidence > 0.7 else (0, 255, 255) if avg_confidence > 0.5 else (0, 0, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                         conf_color, -1)

            # Border
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (255, 255, 255), 2)

            # Text
            conf_text = f"Detection Confidence: {avg_confidence*100:.1f}%"
            text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = bar_x + (bar_width - text_size[0]) // 2
            text_y = bar_y + 20

            cv2.putText(frame, conf_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw confidence indicators for key joints
        if landmarks:
            key_landmarks = [
                (11, "L.Shoulder"),
                (12, "R.Shoulder"),
                (13, "L.Elbow"),
                (14, "R.Elbow"),
                (23, "L.Hip"),
                (24, "R.Hip"),
                (25, "L.Knee"),
                (26, "R.Knee")
            ]

            # Draw confidence dots next to joints
            for idx, name in key_landmarks:
                landmark = landmarks.landmark[idx]
                if landmark.visibility > 0.1:  # Only show if somewhat visible
                    pos_x = int(landmark.x * width)
                    pos_y = int(landmark.y * height)

                    # Offset position slightly
                    indicator_x = pos_x + 15
                    indicator_y = pos_y - 15

                    # Color based on confidence
                    conf_color = (0, 255, 0) if landmark.visibility > 0.7 else (0, 255, 255) if landmark.visibility > 0.5 else (0, 165, 255)

                    # Draw small confidence indicator
                    radius = int(3 + landmark.visibility * 4)
                    cv2.circle(frame, (indicator_x, indicator_y), radius, conf_color, -1)
                    cv2.circle(frame, (indicator_x, indicator_y), radius + 1, (255, 255, 255), 1)

        return frame

    # Draw legend for confidence indicators
    @staticmethod
    def draw_confidence_legend(frame: np.ndarray, position: str = "bottom-left") -> np.ndarray:
        height, width = frame.shape[:2]

        # Legend settings
        legend_width = 220
        legend_height = 110
        padding = 10

        # Calculate position
        if position == "bottom-left":
            x = padding
            y = height - legend_height - padding
        elif position == "top-left":
            x = padding
            y = padding
        elif position == "top-right":
            x = width - legend_width - padding
            y = padding
        else:  # bottom-right
            x = width - legend_width - padding
            y = height - legend_height - padding

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + legend_width, y + legend_height),
                     (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw border
        cv2.rectangle(frame, (x, y), (x + legend_width, y + legend_height),
                     (255, 255, 255), 2)

        # Draw title
        cv2.putText(frame, "Confidence Level", (x + 10, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw indicators
        indicators = [
            ((0, 255, 0), "High (>70%)"),
            ((0, 255, 255), "Medium (50-70%)"),
            ((0, 165, 255), "Low (<50%)")
        ]

        y_offset = y + 45
        for color, label in indicators:
            # Draw color circle
            cv2.circle(frame, (x + 20, y_offset + 8), 6, color, -1)
            cv2.circle(frame, (x + 20, y_offset + 8), 7, (255, 255, 255), 1)

            # Draw label
            cv2.putText(frame, label, (x + 35, y_offset + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            y_offset += 22

        return frame
