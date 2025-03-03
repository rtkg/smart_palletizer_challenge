"""
This script detects boxes based on an RGB-D image using the BoxDetector class.
It allows detection of 'small_box' or 'medium_box' objects and can visualize the detected boxes.

Example usage:
    python src/smart_palletizer/box_detection.py -o small_box -v

Arguments:
    --object, -o: Detection object, can be 'small_box' or 'medium_box'. Default is 'small_box'.
    --visualize, -v: Visualize the detected boxes. This flag is optional.
"""

import os
import argparse
from omegaconf import OmegaConf
from box_detector.box_detector import BoxDetector


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Detect boxes based on an RGB-D image")
    parser.add_argument(
        "--object",
        "-o",
        type=str,
        nargs="?",
        default="small_box",
        help="Detection object, can be 'small_box' or 'medium_box'.",
    )

    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize the detected boxes.")
    args = parser.parse_args()

    # Load configuration
    current_file_path = os.path.abspath(__file__)
    config_path = os.path.join(os.path.dirname(current_file_path), "../../config/box_detector.yaml")
    config = OmegaConf.load(config_path)

    box_detector = BoxDetector(args.object, config)
    box_detector.detect_boxes()
    if args.visualize:
        box_detector.visualize_detected_boxes()
