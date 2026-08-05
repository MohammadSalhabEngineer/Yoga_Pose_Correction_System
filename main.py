# Enhanced Yoga Pose Correction System - MediaPipe-based with hybrid pose comparison

import cv2
import numpy as np
import time
from typing import Optional

from pose_detector import PoseDetector
from angle_calculator import AngleCalculator
from pose_database import PoseDatabase
from pose_comparator import PoseComparator
from feedback_engine import FeedbackEngine
from pose_selector import PoseSelector
from pose_visualizer import PoseVisualizer


# Main application class for yoga pose correction
class YogaPoseCorrector:
    # Initialize the Yoga Pose Corrector
    def __init__(self):
        print("Initializing Yoga Pose Correction System...")

        # Initialize modules
        self.pose_detector = PoseDetector(
            static_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.angle_calculator = AngleCalculator()
        self.pose_database = PoseDatabase(reference_poses_dir="reference_poses")
        self.pose_comparator = PoseComparator(tolerance=30.0)
        self.feedback_engine = FeedbackEngine(
            enable_voice=True,
            feedback_interval=6.0  # Voice feedback every 6 seconds
        )

        # Application state
        self.current_pose = None
        self.current_pose_data = None
        self.reference_image = None
        self.show_reference_overlay = True

        # Performance tracking
        self.frame_count = 0
        self.fps = 0.0
        self.fps_start_time = time.time()

        # Frame skipping for performance
        self.skip_frames = 0
        self.frame_skip_interval = 2  # Process every 3rd frame

        # Zoom settings
        self.zoom_factor = 1.0
        self.min_zoom = 0.3
        self.max_zoom = 3.0
        self.zoom_step = 0.1

        # Last detection cache
        self.last_landmarks = None
        self.last_comparison_result = None

        # Pose completion tracking
        self.pose_correct_start_time = None
        self.pose_correct_duration_threshold = 2.0  # Hold pose correctly for 2 seconds
        self.pose_completed = False

        # Pose change flag to prevent false window closure detection
        self.changing_pose = False

        # Visualization toggles
        self.show_heatmap = True  # Heatmap enabled by default
        self.show_confidence = False  # Confidence disabled by default

        # Pose visualizer
        self.pose_visualizer = PoseVisualizer()

        print("✓ System initialized successfully!")

    # Show pose selection menu and set current pose
    def select_pose(self) -> bool:
        available_poses = self.pose_database.get_available_poses()

        if not available_poses:
            print("No poses available! Please add reference poses.")
            return False

        selector = PoseSelector(available_poses, self.pose_database)
        selected_pose = selector.show_menu()

        if selected_pose is None:
            return False

        # Load selected pose
        self.current_pose = selected_pose
        self.current_pose_data = self.pose_database.get_pose_definition(selected_pose)

        if self.current_pose_data is None:
            print(f"Failed to load pose: {selected_pose}")
            return False

        # Load reference image if available
        ref_data = self.pose_database.get_reference_image(selected_pose)
        if ref_data:
            self.reference_image = ref_data[1]
            print(f"✓ Loaded reference image: {ref_data[0]}")
        else:
            self.reference_image = None
            print("  No reference image available")

        # Update comparator tolerance
        self.pose_comparator.tolerance = self.current_pose_data.get('tolerance', 30.0)

        # Reset pose completion tracking
        self.pose_correct_start_time = None
        self.pose_completed = False

        print(f"\n✓ Loaded pose: {self.current_pose_data['name']}")
        print(f"  Source: {self.current_pose_data.get('source', 'unknown')}")
        print(f"  Tolerance: ±{self.pose_comparator.tolerance}°")
        print(f"  Joints tracked: {len(self.current_pose_data['angles'])}")

        return True

    # Apply zoom to camera frame
    def apply_zoom(self, frame: np.ndarray) -> np.ndarray:
        if self.zoom_factor == 1.0:
            return frame

        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2

        new_width = int(width / self.zoom_factor)
        new_height = int(height / self.zoom_factor)

        x1 = max(0, center_x - new_width // 2)
        y1 = max(0, center_y - new_height // 2)
        x2 = min(width, center_x + new_width // 2)
        y2 = min(height, center_y + new_height // 2)

        cropped = frame[y1:y2, x1:x2]
        zoomed_frame = cv2.resize(cropped, (width, height))

        return zoomed_frame

    # Process a single frame with pose detection and feedback
    def process_frame(self, frame: np.ndarray, should_detect: bool) -> np.ndarray:
        # Detect pose if needed
        if should_detect:
            results = self.pose_detector.detect_pose(frame, convert_color=True)

            if results.pose_landmarks:
                self.last_landmarks = results.pose_landmarks

                # Extract angles
                user_angles = self.angle_calculator.extract_pose_angles(
                    results.pose_landmarks,
                    pose_type="detailed"
                )

                # Compare with reference pose
                self.last_comparison_result = self.pose_comparator.compare_poses(
                    user_angles,
                    self.current_pose_data['angles'],
                    tolerance=self.current_pose_data.get('tolerance')
                )

                # Check if pose is correct and track duration
                if self.last_comparison_result['is_correct']:
                    if self.pose_correct_start_time is None:
                        self.pose_correct_start_time = time.time()
                    else:
                        # Check if pose held correctly long enough
                        hold_duration = time.time() - self.pose_correct_start_time
                        if hold_duration >= self.pose_correct_duration_threshold and not self.pose_completed:
                            self.pose_completed = True
                            # Speak success message
                            self.feedback_engine.speak_success("Perfect! Great job! Do you want to choose a different pose?")
                else:
                    # Reset timer if pose becomes incorrect
                    self.pose_correct_start_time = None

                # Generate voice feedback (only if pose not completed)
                if not self.pose_completed:
                    priority_feedback = self.pose_comparator.get_priority_feedback(
                        self.last_comparison_result,
                        max_feedback=1
                    )
                    self.feedback_engine.speak_feedback(priority_feedback)

        # Draw feedback if we have pose data
        if self.last_landmarks is not None and self.last_comparison_result is not None:
            # Check if pose just completed
            if self.pose_completed:
                # Draw completion overlay
                height, width = frame.shape[:2]
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

                # Draw success message
                success_msg = "PERFECT POSE!"
                text_size = cv2.getTextSize(success_msg, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)[0]
                text_x = (width - text_size[0]) // 2
                text_y = (height // 2) - 60

                cv2.putText(frame, success_msg, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)

                # Draw instruction
                instruction = "Press 'P' to choose another pose"
                inst_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                inst_x = (width - inst_size[0]) // 2
                inst_y = (height // 2) + 20

                cv2.putText(frame, instruction, (inst_x, inst_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # Draw "or ESC to quit"
                quit_msg = "or press 'Q' to quit"
                quit_size = cv2.getTextSize(quit_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                quit_x = (width - quit_size[0]) // 2
                quit_y = (height // 2) + 60

                cv2.putText(frame, quit_msg, (quit_x, quit_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            else:
                # Draw pose skeleton (heatmap or normal)
                if self.show_heatmap:
                    # Draw heatmap skeleton
                    frame = self.pose_visualizer.draw_heatmap_skeleton(
                        frame,
                        self.last_landmarks,
                        self.last_comparison_result['joint_scores'],
                        self.pose_detector.mp_pose,
                        self.pose_detector.mp_drawing
                    )
                    # Draw heatmap legend
                    frame = self.pose_visualizer.draw_heatmap_legend(frame, position="top-right")
                else:
                    # Draw normal skeleton with feedback
                    frame = self.feedback_engine.draw_pose_feedback(
                        frame,
                        self.last_landmarks,
                        self.last_comparison_result,
                        self.pose_detector.mp_pose,
                        self.pose_detector.mp_drawing,
                        self.pose_detector.mp_drawing_styles
                    )

                # Draw confidence visualization if enabled
                if self.show_confidence:
                    frame = self.pose_visualizer.draw_confidence_scores(
                        frame,
                        self.last_landmarks
                    )
                    frame = self.pose_visualizer.draw_confidence_legend(frame, position="bottom-left")

                # Draw score display
                frame = self.feedback_engine.draw_score_display(
                    frame,
                    self.last_comparison_result['similarity_score'],
                    self.last_comparison_result['is_correct']
                )

                # Draw angle display
                frame = self.feedback_engine.draw_angle_display(
                    frame,
                    self.last_comparison_result['joint_scores']
                )

                # Draw feedback messages
                priority_feedback = self.pose_comparator.get_priority_feedback(
                    self.last_comparison_result,
                    max_feedback=3
                )
                frame = self.feedback_engine.draw_feedback_messages(
                    frame,
                    priority_feedback
                )

                # Show progress if pose is currently correct
                if self.last_comparison_result['is_correct'] and self.pose_correct_start_time:
                    hold_duration = time.time() - self.pose_correct_start_time
                    progress = min(hold_duration / self.pose_correct_duration_threshold, 1.0)

                    # Draw progress bar
                    bar_width = 300
                    bar_height = 30
                    bar_x = (frame.shape[1] - bar_width) // 2
                    bar_y = 100

                    # Background
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                                 (50, 50, 50), -1)

                    # Progress
                    progress_width = int(bar_width * progress)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height),
                                 (0, 255, 0), -1)

                    # Border
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                                 (255, 255, 255), 2)

                    # Text
                    cv2.putText(frame, "Hold pose...", (bar_x, bar_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw reference overlay if enabled
        if self.show_reference_overlay and self.reference_image is not None:
            frame = self.feedback_engine.draw_reference_overlay(
                frame,
                self.reference_image,
                alpha=0.3,
                position="bottom-right"
            )

        # Draw UI elements
        frame = self.feedback_engine.draw_ui_elements(
            frame,
            self.current_pose,
            fps=self.fps,
            zoom=self.zoom_factor,
            heatmap_on=self.show_heatmap,
            confidence_on=self.show_confidence
        )

        return frame

    # Detect and let user select camera
    def select_camera(self) -> int:
        print("\n" + "="*60)
        print("Detecting available cameras...")
        print("="*60)

        available_cameras = []

        # Test up to 5 camera indices
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    height, width = frame.shape[:2]
                    available_cameras.append({
                        'index': i,
                        'resolution': f"{width}x{height}"
                    })
                    print(f"  Camera {i}: Found ({width}x{height})")
                cap.release()

        if not available_cameras:
            print("No cameras found!")
            return 0

        if len(available_cameras) == 1:
            print(f"\nUsing Camera {available_cameras[0]['index']}")
            return available_cameras[0]['index']

        # Multiple cameras - let user choose
        print("\nMultiple cameras detected:")
        for cam in available_cameras:
            cam_type = "DroidCam/External" if cam['index'] > 0 else "Built-in Laptop"
            print(f"  {cam['index']}: {cam_type} - {cam['resolution']}")

        print("\nTIP: DroidCam is usually Camera 1 or 2 (higher number)")

        while True:
            try:
                choice = input(f"\nSelect camera (0-{len(available_cameras)-1}) or press ENTER for camera 0: ").strip()

                if choice == "":
                    return 0

                cam_index = int(choice)

                # Check if valid
                if any(cam['index'] == cam_index for cam in available_cameras):
                    print(f"✓ Selected Camera {cam_index}")
                    return cam_index
                else:
                    print(f"Invalid choice. Please select from available cameras.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nUsing default camera 0")
                return 0

    # Main application loop
    def run(self):
        # Select camera first
        camera_index = self.select_camera()

        # Select pose
        if not self.select_pose():
            print("No pose selected. Exiting.")
            return

        # Initialize webcam with selected camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Could not open camera {camera_index}")
            return

        print("\n" + "="*60)
        print("Yoga Pose Corrector Started!")
        print("="*60)
        print(f"Current pose: {self.current_pose_data['name']}")
        print("\nControls:")
        print("  Q or ESC: Quit")
        print("  P: Change pose")
        print("  R: Toggle reference overlay")
        print("  H: Toggle heatmap visualization")
        print("  C: Toggle confidence scores")
        print("  +/-: Zoom in/out")
        print("  0: Reset zoom")
        print("="*60)

        window_name = 'Yoga Pose Corrector'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from webcam")
                    break

                # Frame skipping for performance
                self.skip_frames += 1
                should_detect = (self.skip_frames % self.frame_skip_interval == 0)

                # Resize for faster processing
                height, width = frame.shape[:2]
                if width > 640:
                    scale = 640 / width
                    frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

                # Apply zoom
                frame = self.apply_zoom(frame)

                # Mirror effect
                frame = cv2.flip(frame, 1)

                # Process frame
                frame = self.process_frame(frame, should_detect)

                # Calculate FPS
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    current_time = time.time()
                    self.fps = 30.0 / (current_time - self.fps_start_time)
                    self.fps_start_time = current_time

                # Display frame
                cv2.imshow(window_name, frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == ord('Q') or key == 27:  # Quit
                    print("Quitting application...")
                    break
                elif key == ord('p') or key == ord('P'):  # Change pose
                    print("\nChanging pose...")
                    self.changing_pose = True
                    cv2.destroyWindow(window_name)
                    if not self.select_pose():
                        print("No pose selected, keeping current pose")
                        # Recreate window even if selection failed
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        self.changing_pose = False
                        continue
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    self.changing_pose = False
                    print("\n" + "="*60)
                    print(f"Current pose: {self.current_pose_data['name']}")
                    print("="*60)
                elif key == ord('r') or key == ord('R'):  # Toggle reference
                    self.show_reference_overlay = not self.show_reference_overlay
                    print(f"Reference overlay: {'ON' if self.show_reference_overlay else 'OFF'}")
                elif key == ord('+') or key == ord('='):  # Zoom in
                    self.zoom_factor = min(self.max_zoom, self.zoom_factor + self.zoom_step)
                    print(f"Zoom: {self.zoom_factor:.1f}x")
                elif key == ord('-'):  # Zoom out
                    self.zoom_factor = max(self.min_zoom, self.zoom_factor - self.zoom_step)
                    print(f"Zoom: {self.zoom_factor:.1f}x")
                elif key == ord('0'):  # Reset zoom
                    self.zoom_factor = 1.0
                    print("Zoom reset to 1.0x")
                elif key == ord('h') or key == ord('H'):  # Toggle heatmap
                    self.show_heatmap = not self.show_heatmap
                    print(f"Heatmap: {'ON' if self.show_heatmap else 'OFF'}")
                elif key == ord('c') or key == ord('C'):  # Toggle confidence
                    self.show_confidence = not self.show_confidence
                    print(f"Confidence visualization: {'ON' if self.show_confidence else 'OFF'}")

                # Check if window was closed (but not during pose change)
                if not self.changing_pose:
                    try:
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            print("Window closed, quitting...")
                            break
                    except cv2.error:
                        # Window doesn't exist yet, ignore
                        pass

        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected, quitting...")
        except Exception as e:
            print(f"Error occurred: {e}")
            import traceback
            traceback.print_exc()

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.cleanup()

    # Release all resources
    def cleanup(self):
        print("Cleaning up resources...")
        self.pose_detector.cleanup()
        self.pose_database.cleanup()
        self.feedback_engine.cleanup()
        print("✓ Cleanup complete")


# Main entry point
def main():
    print("\n" + "="*60)
    print("  Yoga Pose Correction System v2.0")
    print("  Hybrid approach: Hardcoded + Image-based comparison")
    print("="*60 + "\n")

    corrector = YogaPoseCorrector()
    corrector.run()


if __name__ == "__main__":
    main()
