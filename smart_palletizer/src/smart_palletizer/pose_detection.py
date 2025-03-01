from pose_detector.pose_detector import PoseDetector
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Detect box boses based on and RGB-D image ")
    parser.add_argument(
        "--object",
        type=str,
        nargs="?",
        default="small_box",
        help="Detection object, can be 'small_box' or 'medium_box'.",
    )

    parser.add_argument("--visualize", action="store_false", help="Visualize the point cloud with detected planes.")
    args = parser.parse_args()

    pose_detector = PoseDetector(args.object)
    pose_detector.detect_poses()
    if args.visualize:
        pose_detector.visualize_detected_poses()
