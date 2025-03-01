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


class BoxDetector:
    def __init__(self, object):
        self.set_detection_object(object)
        self.detected_boxes = None

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

    def detect_boxes(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device=device)

        color_image_pil = Image.fromarray(cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB))
        masks = generator(color_image_pil, points_per_batch=128, pred_iou_thresh=0.99)

        box_candidates = self.filter_masks(masks["masks"])
        self.detected_boxes = self.find_boxes(box_candidates)

        with open(os.path.join(self.data_path, "detected_boxes.pkl"), "wb") as f:
            pickle.dump(self.detected_boxes, f)

    def find_boxes(self, box_candidates):
        detected_boxes = []

        eps = 0.2
        fx = self.intrinsics[0, 0]
        fy = self.intrinsics[1, 1]
        permutations = list(itertools.permutations(self.box_dimensions, 2))
        for box_candidate in box_candidates:
            contours, _ = cv2.findContours(box_candidate.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bbox_contour = cv2.minAreaRect(contours[0])
            bbox = cv2.boxPoints(bbox_contour)
            w = np.linalg.norm(bbox[0] - bbox[1])
            h = np.linalg.norm(bbox[1] - bbox[2])
            z3d = np.median(self.depth_image[box_candidate]) * self.depth_scale
            if z3d > 1.9:
                continue
            for perm in permutations:
                # camera projection using the intrinsics
                w_box = perm[0] * fx / z3d
                h_box = perm[1] * fy / z3d

                if abs((w_box - w) / w_box) < eps and abs((h_box - h) / h_box) < eps:
                    detected_boxes.append((box_candidate, bbox.astype(np.int32)))
                    break
        return detected_boxes

    def visualize_detected_boxes(self):
        if not self.detected_boxes:
            print("No boxes detected.")
            return

        # Visualize the bounding box and box_candidate
        color_image = self.color_image.copy()
        for box, bbox in self.detected_boxes:
            cv2.drawContours(color_image, [bbox], 0, (255, 0, 0), 2)
            plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
            plt.contour(box, colors="r")

        plt.axis("off")
        plt.show()

    def filter_masks(self, masks):
        filtered_masks = []

        for i, mask in enumerate(masks):
            # filter by containment
            is_contained = False
            for j, mask_j in enumerate(masks):
                # check if current mask is fully contained in any other mask
                if i == j:
                    continue

                if np.all(mask_j & mask == mask):
                    is_contained = True
                    break

            if is_contained:
                continue

            filtered_masks.append(mask)

        return filtered_masks
