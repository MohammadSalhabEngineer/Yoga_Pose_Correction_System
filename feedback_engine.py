# Feedback engine for providing visual and voice feedback

import cv2
import pyttsx3
import threading
import time
import numpy as np
from typing import List, Dict, Optional, Tuple


# Handles visual and voice feedback for pose correction
class FeedbackEngine:
    # Initialize feedback engine with voice and visual settings
    def __init__(self,
                 enable_voice: bool = True,
                 voice_rate: int = 180,
                 voice_volume: float = 0.9,
                 feedback_interval: float = 6.0):
        self.enable_voice = enable_voice
        self.feedback_interval = feedback_interval
        self.last_feedback_time = 0
        self.tts_speaking = False

        # Colors for feedback
        self.correct_color = (0, 255, 0)  # Green
        self.incorrect_color = (0, 0, 255)  # Red
        self.neutral_color = (255, 255, 255)  # White
        self.warning_color = (0, 255, 255)  # Yellow

        # Initialize TTS if enabled
        if self.enable_voice:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', voice_rate)
                self.tts_engine.setProperty('volume', voice_volume)

                # Try to set a female voice
                voices = self.tts_engine.getProperty('voices')
                for voice in voices:
                    # Look for female voice (usually contains 'female' or 'zira' on Windows)
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        print(f"Using voice: {voice.name}")
                        break
                    elif 'david' not in voice.name.lower() and len(voices) > 1:
                        # On some systems, second voice is female
                        self.tts_engine.setProperty('voice', voice.id)
                        print(f"Using voice: {voice.name}")
                        break
            except Exception as e:
                print(f"Warning: Failed to initialize TTS: {e}")
                self.enable_voice = False

    # Speak feedback messages using TTS (non-blocking)
    def speak_feedback(self, messages: List[str], max_messages: int = 1):
        if not self.enable_voice or not messages or self.tts_speaking:
            return

        current_time = time.time()
        if current_time - self.last_feedback_time < self.feedback_interval:
            return

        self.tts_speaking = True
        self.last_feedback_time = current_time

        # Limit messages
        messages_to_speak = messages[:max_messages]
        message = ". ".join(messages_to_speak)

        def speak():
            try:
                self.tts_engine.say(message)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")
            finally:
                self.tts_speaking = False

        tts_thread = threading.Thread(target=speak)
        tts_thread.daemon = True
        tts_thread.start()

    # Speak a success message immediately (bypasses interval check)
    def speak_success(self, message: str = "Perfect! Great job!"):
        if not self.enable_voice or self.tts_speaking:
            return

        self.tts_speaking = True

        def speak():
            try:
                self.tts_engine.say(message)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")
            finally:
                self.tts_speaking = False

        tts_thread = threading.Thread(target=speak)
        tts_thread.daemon = True
        tts_thread.start()

    def draw_pose_feedback(self,
                          image: np.ndarray,
                          landmarks,
                          comparison_result: Dict[str, any],
                          mp_pose,
                          mp_drawing,
                          mp_drawing_styles) -> np.ndarray:
        # Draw pose skeleton with color-coded feedback
        if landmarks is None:
            return image

        annotated_image = image.copy()

        # Draw pose skeleton
        mp_drawing.draw_landmarks(
            annotated_image,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # Draw joint status indicators on LEFT side only
        y_offset = 100  # Start lower to avoid overlap with pose name
        joint_scores = comparison_result.get('joint_scores', {})

        for joint, data in joint_scores.items():
            color = self.correct_color if data['within_tolerance'] else self.incorrect_color
            joint_display = joint.replace('_', ' ').title()
            status = 'OK' if data['within_tolerance'] else 'Adjust'
            text = f"{joint_display}: {status}"

            cv2.putText(annotated_image, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_offset += 22

        return annotated_image

    def draw_score_display(self,
                          image: np.ndarray,
                          similarity_score: float,
                          is_correct: bool) -> np.ndarray: # Draw overall score and status
        height, width = image.shape[:2]

        # Determine color based on score
        if similarity_score >= 80:
            color = self.correct_color
        elif similarity_score >= 60:
            color = self.warning_color
        else:
            color = self.incorrect_color

        # Draw score box in top-right corner
        box_width = 200
        box_height = 80
        box_x = width - box_width - 10
        box_y = 10

        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (box_x, box_y),
                     (box_x + box_width, box_y + box_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        # Draw score text
        score_text = f"Score: {similarity_score:.1f}%"
        cv2.putText(image, score_text, (box_x + 10, box_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw status
        status_text = "CORRECT!" if is_correct else "ADJUST"
        cv2.putText(image, status_text, (box_x + 10, box_y + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return image

    def draw_feedback_messages(self,
                              image: np.ndarray,
                              messages: List[str],
                              max_messages: int = 3) -> np.ndarray: # Draw feedback messages
        if not messages:
            # Show success message
            cv2.putText(image, "Perfect Pose!", (10, image.shape[0] - 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.correct_color, 2)
            return image

        height = image.shape[0]
        y_start = height - 120

        for i, message in enumerate(messages[:max_messages]):
            y_pos = y_start + i * 30
            cv2.putText(image, message, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.incorrect_color, 2)

        return image

    def draw_angle_display(self,
                          image: np.ndarray,
                          joint_scores: Dict[str, Dict]) -> np.ndarray:
        """
        Display current angles with comparison to reference.

        Args:
            image: Input image
            joint_scores: Joint scores from comparison result

        Returns:
            Image with angle display
        """
        height, width = image.shape[:2]
        x_offset = width - 280
        y_offset = 120

        cv2.putText(image, "Angles:", (x_offset, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.neutral_color, 2)
        y_offset += 25

        for joint, data in joint_scores.items():
            color = self.correct_color if data['within_tolerance'] else self.incorrect_color
            joint_display = joint.replace('_', ' ').title()

            # Show user angle vs reference (cleaner format)
            angle_text = f"{joint_display}: {data['user_angle']:.0f} ({data['reference_angle']:.0f})"
            cv2.putText(image, angle_text, (x_offset, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y_offset += 20

        return image

    def draw_reference_overlay(self,
                              image: np.ndarray,
                              reference_image: Optional[np.ndarray],
                              alpha: float = 0.3,
                              position: str = "bottom-left") -> np.ndarray:
        """
        Draw semi-transparent reference pose overlay.

        Args:
            image: Input image
            reference_image: Reference pose image
            alpha: Transparency (0.0-1.0)
            position: Where to place overlay ("bottom-left", "bottom-right", "top-left", "top-right")

        Returns:
            Image with overlay
        """
        if reference_image is None:
            return image

        height, width = image.shape[:2]
        ref_height, ref_width = reference_image.shape[:2]

        # Resize reference image to fit in corner (1/4 of screen)
        max_size = min(width // 3, height // 3)
        scale = min(max_size / ref_width, max_size / ref_height)
        new_width = int(ref_width * scale)
        new_height = int(ref_height * scale)

        resized_ref = cv2.resize(reference_image, (new_width, new_height))

        # Determine position
        if position == "bottom-left":
            y_start = height - new_height - 10
            x_start = 10
        elif position == "bottom-right":
            y_start = height - new_height - 10
            x_start = width - new_width - 10
        elif position == "top-left":
            y_start = 10
            x_start = 10
        else:  # top-right
            y_start = 10
            x_start = width - new_width - 10

        # Blend reference image
        roi = image[y_start:y_start+new_height, x_start:x_start+new_width]
        blended = cv2.addWeighted(roi, 1-alpha, resized_ref, alpha, 0)
        image[y_start:y_start+new_height, x_start:x_start+new_width] = blended

        # Add label
        cv2.putText(image, "Reference", (x_start + 5, y_start + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.neutral_color, 1)

        return image

    def draw_ui_elements(self,
                        image: np.ndarray,
                        pose_name: str,
                        fps: Optional[float] = None,
                        zoom: Optional[float] = None,
                        heatmap_on: bool = False,
                        confidence_on: bool = False) -> np.ndarray: # Draw UI elements
        height, width = image.shape[:2]

        # Pose name at top - centered and prominent
        pose_display = pose_name.replace('_', ' ')
        text_size = cv2.getTextSize(pose_display, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (width - text_size[0]) // 2

        # Black background for better visibility
        cv2.rectangle(image, (text_x - 10, 5), (text_x + text_size[0] + 10, 40), (0, 0, 0), -1)
        cv2.putText(image, pose_display, (text_x, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # FPS counter and visualization status (top left, small)
        y_offset = 60
        if fps is not None:
            cv2.putText(image, f"FPS: {fps:.0f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.neutral_color, 1)
            y_offset += 20

        # Show visualization status
        if heatmap_on:
            cv2.putText(image, "Heatmap: ON", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            y_offset += 20

        if confidence_on:
            cv2.putText(image, "Confidence: ON", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Controls hint at bottom - centered
        controls_text = "Q:Quit | P:Pose | R:Ref | H:Heatmap | C:Conf | +/-:Zoom"
        controls_size = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        controls_x = (width - controls_size[0]) // 2

        cv2.putText(image, controls_text,
                   (controls_x, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   self.neutral_color, 1)

        return image

    def cleanup(self): # Release resources
        if self.enable_voice:
            try:
                self.tts_engine.stop()
            except:
                pass
