# Pose database for managing yoga poses and their reference data

import os
import cv2
import json
from typing import Dict, List, Optional, Tuple
from angle_calculator import AngleCalculator
from pose_detector import PoseDetector


# Manages yoga pose definitions and reference data
class PoseDatabase:
    # Hardcoded pose definitions for accuracy
    HARDCODED_POSES = {
        'TADASANA_Mountain_Arms_Up': {
            'name': 'TADASANA - Mountain Arms Up',
            'description': 'Stand tall with arms raised straight above head',
            'angles': {
                'left_elbow': 180.0,     # Arms straight
                'right_elbow': 180.0,    # Arms straight
                'left_shoulder': 160.0,  # Arms raised up
                'right_shoulder': 160.0, # Arms raised up
                'torso': 180.0,          # Straight back
                'left_knee': 180.0,      # Legs straight
                'right_knee': 180.0      # Legs straight
            },
            'tolerance': 30.0,
            'reference_image': 'TADASANA_Mountain_Arms_Up.jpg'
        },
        'Virabhadrasana_Warrior_2': {
            'name': 'VIRABHADRASANA II - Warrior 2',
            'description': 'Warrior pose with arms extended to sides and front knee bent',
            'angles': {
                'left_elbow': 180.0,     # Arms straight out to sides
                'right_elbow': 180.0,    # Arms straight out to sides
                'left_shoulder': 90.0,   # Left arm extended to side (90 degrees from torso)
                'right_shoulder': 90.0,  # Right arm extended to side (90 degrees from torso)
                'torso': 180.0,          # Upper body upright
                'left_knee': 90.0,       # Front knee bent at 90 degrees
                'right_knee': 180.0      # Back leg straight
            },
            'tolerance': 30.0,
            'reference_image': 'Virabhadrasana_Warrior_2.jpg'
        },
        'Vrksasana_Tree_Pose': {
            'name': 'VRKSASANA - Tree Pose',
            'description': 'Balance on one leg with hands in prayer position at chest',
            'angles': {
                'left_elbow': 90.0,      # Elbows bent, hands at chest in prayer
                'right_elbow': 90.0,     # Elbows bent, hands at chest in prayer
                'left_shoulder': 90.0,   # Arms at chest level
                'right_shoulder': 90.0,  # Arms at chest level
                'torso': 180.0,          # Straight upright posture
                'left_knee': 90.0,       # Bent leg (foot on inner thigh)
                'right_knee': 180.0      # Standing leg straight
            },
            'tolerance': 30.0,
            'reference_image': 'Vrksasana_Tree_Pose.jpg'
        },
        'Standing_Side_Bend_Pose': {
            'name': 'STANDING SIDE BEND',
            'description': 'Standing with side bend, one arm reaching over',
            'angles': {
                'left_elbow': 180.0,     # Reaching arm straight
                'right_elbow': 180.0,    # Down arm straight
                'left_shoulder': 140.0,  # Reaching arm extended up and over
                'right_shoulder': 120.0, # Down arm along body
                'torso': 140.0,          # Torso bent to side (not straight)
                'left_knee': 180.0,      # Both legs straight
                'right_knee': 180.0      # Both legs straight
            },
            'tolerance': 30.0,
            'reference_image': 'Standing_Side_Bend_Pose.jpg'
        }
    }

    # Initialize pose database with reference poses directory
    def __init__(self, reference_poses_dir: str = "reference_poses"):
        self.reference_poses_dir = reference_poses_dir
        self.pose_detector = PoseDetector(static_mode=True, model_complexity=1)
        self.image_based_poses = {}
        self.cache_file = os.path.join(reference_poses_dir, "pose_cache.json")

        # Load cached pose data if available
        self._load_cache()

    # Get list of all available poses (hardcoded + image-based)
    def get_available_poses(self) -> List[str]:
        available = list(self.HARDCODED_POSES.keys())

        # Add image-based poses
        if os.path.exists(self.reference_poses_dir):
            for filename in os.listdir(self.reference_poses_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    pose_name = os.path.splitext(filename)[0]
                    if pose_name not in available:
                        available.append(pose_name)

        return sorted(available)

    # Get complete pose definition including angles
    def get_pose_definition(self, pose_name: str) -> Optional[Dict]:
        # Check hardcoded poses first
        if pose_name in self.HARDCODED_POSES:
            pose_data = self.HARDCODED_POSES[pose_name].copy()
            pose_data['source'] = 'hardcoded'
            return pose_data

        # Check image-based poses
        if pose_name in self.image_based_poses:
            return self.image_based_poses[pose_name]

        # Try to load from image
        pose_data = self._load_pose_from_image(pose_name)
        if pose_data:
            self.image_based_poses[pose_name] = pose_data
            self._save_cache()
            return pose_data

        return None

    # Load pose data from reference image using MediaPipe
    def _load_pose_from_image(self, pose_name: str) -> Optional[Dict]:
        # Look for image file
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            potential_path = os.path.join(self.reference_poses_dir, pose_name + ext)
            if os.path.exists(potential_path):
                image_path = potential_path
                break

        if not image_path:
            return None

        # Load and process image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None

        # Detect pose in image
        results = self.pose_detector.detect_pose(image)
        if not results.pose_landmarks:
            print(f"No pose detected in reference image: {image_path}")
            return None

        # Extract angles
        angles = AngleCalculator.extract_pose_angles(results.pose_landmarks, pose_type="detailed")

        pose_data = {
            'name': pose_name.replace('_', ' ').title(),
            'description': f'Reference pose from {os.path.basename(image_path)}',
            'angles': angles,
            'tolerance': 30.0,
            'source': 'image',
            'image_path': image_path
        }

        return pose_data

    # Get reference image for a pose
    def get_reference_image(self, pose_name: str) -> Optional[Tuple[str, any]]:
        # Look for image file
        for ext in ['.jpg', '.jpeg', '.png']:
            image_path = os.path.join(self.reference_poses_dir, pose_name + ext)
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    return (image_path, image)

        return None

    # Load cached pose data from JSON file
    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.image_based_poses = json.load(f)
                print(f"Loaded {len(self.image_based_poses)} cached poses")
            except Exception as e:
                print(f"Failed to load pose cache: {e}")
                self.image_based_poses = {}

    # Save pose data to cache file
    def _save_cache(self):
        try:
            os.makedirs(self.reference_poses_dir, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.image_based_poses, f, indent=2)
            print(f"Saved {len(self.image_based_poses)} poses to cache")
        except Exception as e:
            print(f"Failed to save pose cache: {e}")

    # Add a custom pose definition
    def add_custom_pose(self, pose_name: str, angles: Dict[str, float],
                       tolerance: float = 30.0, description: str = ""):
        self.image_based_poses[pose_name] = {
            'name': pose_name.replace('_', ' ').title(),
            'description': description or f'Custom pose: {pose_name}',
            'angles': angles,
            'tolerance': tolerance,
            'source': 'custom'
        }
        self._save_cache()

    # Release resources
    def cleanup(self):
        self.pose_detector.cleanup()
