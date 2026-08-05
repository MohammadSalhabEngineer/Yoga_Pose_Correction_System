# Pose comparison module with hybrid approach (hardcoded + image-based)

from typing import Dict, List, Tuple, Optional
from angle_calculator import AngleCalculator


# Compares user poses with reference poses using hybrid approach
class PoseComparator:
    # Initialize pose comparator with default tolerance
    def __init__(self, tolerance: float = 30.0):
        self.tolerance = tolerance
        self.angle_calculator = AngleCalculator()

    # Compare user pose with reference pose and return detailed results
    def compare_poses(self,
                     user_angles: Dict[str, float],
                     reference_angles: Dict[str, float],
                     tolerance: Optional[float] = None) -> Dict[str, any]:
        if tolerance is None:
            tolerance = self.tolerance

        joint_scores = {}
        feedback = []
        total_score = 0.0
        joint_count = 0

        for joint, ref_angle in reference_angles.items():
            if joint not in user_angles:
                continue

            user_angle = user_angles[joint]
            diff = abs(user_angle - ref_angle)

            # Calculate score for this joint
            if diff <= tolerance:
                joint_score = 100.0 * (1.0 - diff / tolerance)
            else:
                joint_score = max(0.0, 100.0 * (1.0 - diff / 180.0))

            joint_scores[joint] = {
                'score': joint_score,
                'user_angle': user_angle,
                'reference_angle': ref_angle,
                'difference': diff,
                'within_tolerance': diff <= tolerance
            }

            total_score += joint_score
            joint_count += 1

            # Generate feedback if not within tolerance
            if diff > tolerance:
                feedback_msg = self._generate_joint_feedback(joint, user_angle, ref_angle, tolerance)
                if feedback_msg:
                    feedback.append(feedback_msg)

        overall_score = total_score / joint_count if joint_count > 0 else 0.0
        is_correct = overall_score >= 70.0  # 70% threshold for correct pose

        return {
            'similarity_score': overall_score,
            'joint_scores': joint_scores,
            'feedback': feedback,
            'is_correct': is_correct,
            'joints_analyzed': joint_count
        }

    # Generate specific feedback for a joint
    def _generate_joint_feedback(self,
                                 joint: str,
                                 user_angle: float,
                                 ref_angle: float,
                                 tolerance: float) -> str:
        diff = user_angle - ref_angle

        # Define feedback templates for different joints
        feedback_templates = {
            'left_elbow': {
                'too_low': "Straighten your left arm more",
                'too_high': "Bend your left arm slightly"
            },
            'right_elbow': {
                'too_low': "Straighten your right arm more",
                'too_high': "Bend your right arm slightly"
            },
            'left_shoulder': {
                'too_low': "Raise your left arm higher",
                'too_high': "Lower your left arm slightly"
            },
            'right_shoulder': {
                'too_low': "Raise your right arm higher",
                'too_high': "Lower your right arm slightly"
            },
            'left_knee': {
                'too_low': "Bend your left knee more",
                'too_high': "Straighten your left knee more"
            },
            'right_knee': {
                'too_low': "Bend your right knee more",
                'too_high': "Straighten your right knee more"
            },
            'torso': {
                'too_low': "Straighten your back more",
                'too_high': "Relax your back slightly"
            },
            'left_hip': {
                'too_low': "Adjust your left hip position",
                'too_high': "Adjust your left hip position"
            },
            'right_hip': {
                'too_low': "Adjust your right hip position",
                'too_high': "Adjust your right hip position"
            }
        }

        if joint not in feedback_templates:
            return f"Adjust your {joint.replace('_', ' ')}"

        template = feedback_templates[joint]

        if diff < -tolerance:
            return template['too_low']
        elif diff > tolerance:
            return template['too_high']

        return ""

    # Get prioritized feedback messages (worst joints first)
    def get_priority_feedback(self,
                             comparison_result: Dict[str, any],
                             max_feedback: int = 3) -> List[str]:
        if comparison_result['is_correct']:
            return []

        # Sort joints by score (lowest first)
        joint_scores = comparison_result['joint_scores']
        sorted_joints = sorted(
            joint_scores.items(),
            key=lambda x: x[1]['score']
        )

        # Get feedback from comparison result
        all_feedback = comparison_result['feedback']

        # If we have fewer feedback than max, return all
        if len(all_feedback) <= max_feedback:
            return all_feedback

        # Otherwise, prioritize based on joint scores
        priority_feedback = []
        for joint, _ in sorted_joints:
            for feedback in all_feedback:
                if joint.replace('_', ' ') in feedback.lower():
                    if feedback not in priority_feedback:
                        priority_feedback.append(feedback)
                        if len(priority_feedback) >= max_feedback:
                            return priority_feedback

        return priority_feedback[:max_feedback]

    # Get detailed text analysis of pose comparison
    def get_detailed_analysis(self, comparison_result: Dict[str, any]) -> str:
        lines = [
            f"Overall Score: {comparison_result['similarity_score']:.1f}%",
            f"Status: {'CORRECT' if comparison_result['is_correct'] else 'NEEDS ADJUSTMENT'}",
            f"Joints Analyzed: {comparison_result['joints_analyzed']}",
            "",
            "Joint Details:"
        ]

        for joint, data in comparison_result['joint_scores'].items():
            status = "✓" if data['within_tolerance'] else "✗"
            joint_name = joint.replace('_', ' ').title()
            lines.append(
                f"  {status} {joint_name}: {data['score']:.1f}% "
                f"(User: {data['user_angle']:.1f}°, Ref: {data['reference_angle']:.1f}°)"
            )

        if comparison_result['feedback']:
            lines.append("")
            lines.append("Feedback:")
            for feedback in comparison_result['feedback']:
                lines.append(f"  • {feedback}")

        return "\n".join(lines)
