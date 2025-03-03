"""
This script runs the box pose detector using an RGB-D image.
It allows detection of 'small_box' or 'medium_box' objects and can visualize the detected poses.

Example usage:
    python src/smart_palletizer/pose_detection.py -o small_box -v

Arguments:
    --object, -o: Detection object, can be 'small_box' or 'medium_box'. Default is 'small_box'.
    --visualize, -v: Visualize the point cloud with detected planes. This flag is optional.
"""

import os
import argparse
from pose_detector.pose_detector import PoseDetector
from omegaconf import OmegaConf


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Detect box poses based on an RGB-D image")
    parser.add_argument(
        "--object",
        "-o",
        type=str,
        nargs="?",
        default="small_box",
        help="Detection object, can be 'small_box' or 'medium_box'.",
    )

    parser.add_argument(
        "--visualize", "-v", action="store_true", help="Visualize the point cloud with detected planes."
    )
    args = parser.parse_args()

    # Load configuration
    current_file_path = os.path.abspath(__file__)
    config_path = os.path.join(os.path.dirname(current_file_path), "../../config/pose_detector.yaml")
    config = OmegaConf.load(config_path)

    pose_detector = PoseDetector(args.object, config)
    pose_detector.detect_poses()
    if args.visualize:
        pose_detector.visualize_detected_poses()
