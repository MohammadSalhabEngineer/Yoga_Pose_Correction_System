# Interactive pose selection menu

import cv2
import numpy as np
from typing import List, Optional


# Interactive menu for selecting yoga poses
class PoseSelector:
    # Initialize pose selector with available poses and database
    def __init__(self, available_poses: List[str], pose_database):
        self.available_poses = available_poses
        self.pose_database = pose_database
        self.selected_index = 0

    # Show interactive pose selection menu
    def show_menu(self, window_name: str = "Select Pose") -> Optional[str]:
        if not self.available_poses:
            print("No poses available!")
            return None

        print("\n" + "="*60)
        print("Yoga Pose Selection Menu")
        print("="*60)
        print("\nAvailable Poses:")

        for i, pose_name in enumerate(self.available_poses):
            pose_def = self.pose_database.get_pose_definition(pose_name)
            display_name = pose_def['name'] if pose_def else pose_name
            source = pose_def.get('source', 'unknown') if pose_def else 'unknown'
            print(f"  {i+1}. {display_name} [{source}]")

        print("\nControls:")
        print("  ↑/↓ or W/S: Navigate")
        print("  ENTER or SPACE: Select pose")
        print("  Q or ESC: Quit")
        print("="*60)

        # Create visualization window
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            # If camera not available, use text-based selection
            return self._text_based_selection()

        selected_pose = None

        while True:
            # Create menu image
            menu_image = self._create_menu_image()

            # Get camera frame for preview
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                # Resize camera preview
                preview_height = 300
                aspect_ratio = frame.shape[1] / frame.shape[0]
                preview_width = int(preview_height * aspect_ratio)
                frame_preview = cv2.resize(frame, (preview_width, preview_height))

                # Combine menu and preview
                combined = np.zeros((max(menu_image.shape[0], preview_height),
                                   menu_image.shape[1] + preview_width, 3), dtype=np.uint8)
                combined[:menu_image.shape[0], :menu_image.shape[1]] = menu_image
                combined[:preview_height, menu_image.shape[1]:] = frame_preview

                # Add preview label
                cv2.putText(combined, "Camera Preview",
                           (menu_image.shape[1] + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                display_image = combined
            else:
                display_image = menu_image

            cv2.imshow(window_name, display_image)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                print("Selection cancelled")
                break
            elif key == ord('w') or key == ord('W') or key == 82:  # W or Up arrow
                self.selected_index = (self.selected_index - 1) % len(self.available_poses)
            elif key == ord('s') or key == ord('S') or key == 83:  # S or Down arrow
                self.selected_index = (self.selected_index + 1) % len(self.available_poses)
            elif key == 13 or key == 32:  # ENTER or SPACE
                selected_pose = self.available_poses[self.selected_index]
                print(f"Selected: {selected_pose}")
                break

        cap.release()
        cv2.destroyWindow(window_name)

        return selected_pose

    # Create menu visualization image
    def _create_menu_image(self) -> np.ndarray:
        # Image dimensions
        width = 600
        item_height = 60
        header_height = 80
        footer_height = 60
        height = header_height + len(self.available_poses) * item_height + footer_height

        # Create black background
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw header
        cv2.rectangle(image, (0, 0), (width, header_height), (40, 40, 40), -1)
        cv2.putText(image, "Select Your Yoga Pose",
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # Draw pose items
        y_offset = header_height

        for i, pose_name in enumerate(self.available_poses):
            is_selected = (i == self.selected_index)

            # Background color
            if is_selected:
                color = (60, 120, 60)  # Green for selected
            else:
                color = (30, 30, 30)  # Dark gray for unselected

            # Draw item background
            cv2.rectangle(image,
                         (10, y_offset + 5),
                         (width - 10, y_offset + item_height - 5),
                         color, -1)

            # Get pose info
            pose_def = self.pose_database.get_pose_definition(pose_name)
            display_name = pose_def['name'] if pose_def else pose_name
            description = pose_def.get('description', 'No description') if pose_def else 'No description'
            source = pose_def.get('source', 'unknown') if pose_def else 'unknown'

            # Draw selection indicator
            if is_selected:
                cv2.putText(image, ">",
                           (20, y_offset + 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Draw pose name
            text_x = 50 if is_selected else 30
            cv2.putText(image, display_name,
                       (text_x, y_offset + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Draw source tag
            source_color = (100, 255, 100) if source == 'hardcoded' else (255, 255, 100)
            cv2.putText(image, f"[{source}]",
                       (text_x, y_offset + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, source_color, 1)

            y_offset += item_height

        # Draw footer with controls
        cv2.rectangle(image, (0, height - footer_height), (width, height), (40, 40, 40), -1)
        cv2.putText(image, "W/S: Navigate | ENTER: Select | Q: Quit",
                   (20, height - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return image

    # Fallback text-based selection if camera not available
    def _text_based_selection(self) -> Optional[str]:
        while True:
            try:
                choice = input("\nEnter pose number (or 'q' to quit): ").strip()

                if choice.lower() == 'q':
                    return None

                index = int(choice) - 1
                if 0 <= index < len(self.available_poses):
                    return self.available_poses[index]
                else:
                    print(f"Invalid choice. Please enter 1-{len(self.available_poses)}")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nSelection cancelled")
                return None
