import os
import itertools
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline
import pickle
from utils.data_loading import load_image, load_camera_parameters
from utils.visualization import show_mask


class PoseDetector:
    def __init__(self, object):
        self.set_detection_object(object)
        self.detected_poses = None

    def set_detection_object(self, object):
        current_file_path = os.path.abspath(__file__)
        data_path = os.path.join(os.path.dirname(current_file_path), "../../../data/", object)

        color_image_path = os.path.join(data_path, "color_image.png")
        depth_image_path = os.path.join(data_path, "raw_depth.png")
        intrinsics_path = os.path.join(data_path, "intrinsics.json")
        extrinsics_path = os.path.join(data_path, "cam2root.json")

        self.color_image = load_image(color_image_path)
        self.depth_image = load_image(depth_image_path, is_depth_image=True)
        self.width, self.height, self.intrinsics, self.extrinsics = load_camera_parameters(
            intrinsics_path, extrinsics_path
        )
        self.depth_scale = 1e-3  # assuming depth measurements to be given in [mm]
        self.data_path = data_path
        self.object = object
        if object == "small_box":
            self.box_dimensions = [0.255, 0.155, 0.100]
        elif object == "medium_box":
            self.box_dimensions = [0.340, 0.250, 0.095]
        else:
            raise ValueError("Invalid object type. Supported objects are 'small_box' and 'medium_box'.")

    def detect_poses(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device=device)

        color_image_pil = Image.fromarray(cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB))
        masks = generator(color_image_pil, points_per_batch=128, pred_iou_thresh=0.99)

        with open(os.path.join(self.data_path, "detected_poses.pkl"), "wb") as f:
            pickle.dump(self.detected_poses, f)

    def visualize_detected_poses(self):
        if not self.detected_poses:
            print("No poses detected.")
            return

        # Visualize the bounding box and box_candidate
        color_image = self.color_image.copy()
        for box, bbox in self.detected_poses:
            cv2.drawContours(color_image, [bbox], 0, (255, 0, 0), 2)
            plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
            plt.contour(box, colors="r")

        plt.axis("off")
        plt.show()
