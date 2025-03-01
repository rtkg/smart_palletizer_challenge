from box_detector.box_detector import BoxDetector
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Detect boxes based on and RGB-D image ")
    parser.add_argument(
        "--object",
        type=str,
        nargs="?",
        default="small_box",
        help="Detection object, can be 'small_box' or 'medium_box'.",
    )

    parser.add_argument("--visualize", action="store_false", help="Visualize the point cloud with detected planes.")
    args = parser.parse_args()

    box_detector = BoxDetector(args.object)
    box_detector.detect_boxes()
    if args.visualize:
        box_detector.visualize_detected_boxes()
