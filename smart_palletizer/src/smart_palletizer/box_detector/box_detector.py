import os
import pickle
from typing import List, Tuple
import itertools
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline
from utils.data_loading import PalletizerData
from omegaconf import DictConfig


class BoxDetector:
    """
    A class to detect boxes in RGB-D images.

    Attributes:
        data (PalletizerData): The loaded data for the palletizer.
        detected_boxes (List[Tuple[np.ndarray, np.ndarray]]): The detected boxes and their bounding boxes.

    Methods:
        detect_boxes(): Detect boxes in the RGB-D image.
        visualize_detected_boxes(): Visualize the detected boxes.
    """

    def __init__(self, object: str, config: DictConfig) -> None:
        """
        Initialize the BoxDetector with the given object and configuration.

        Args:
            object (str): The object to detect boxes for.
            config (DictConfig): The configuration dictionary.
        """
        self.data = PalletizerData(config.palletizer_data, object)
        self.detected_boxes = None

    def detect_boxes(self) -> None:
        """
        Detect boxes in the RGB-D image.

        Returns:
            None
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device=device)

        color_image_pil = Image.fromarray(cv2.cvtColor(self.data.color_image, cv2.COLOR_BGR2RGB))
        masks = generator(color_image_pil, points_per_batch=128, pred_iou_thresh=0.99)

        box_candidates = self._filter_masks(masks["masks"])
        self.detected_boxes = self._find_boxes(box_candidates)

        with open(os.path.join(self.data.data_path, "detected_boxes.pkl"), "wb") as f:
            pickle.dump(self.detected_boxes, f)

    def _find_boxes(self, box_candidates: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Find boxes from the list of box candidates.

        Args:
            box_candidates (List[np.ndarray]): List of box candidate masks.

        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: List of detected boxes and their bounding boxes.
        """
        detected_boxes = []

        eps = 0.2
        fx = self.data.intrinsics[0, 0]
        fy = self.data.intrinsics[1, 1]
        permutations = list(itertools.permutations(self.data.box_dimensions, 2))
        for box_candidate in box_candidates:
            contours, _ = cv2.findContours(box_candidate.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bbox_contour = cv2.minAreaRect(contours[0])
            bbox = cv2.boxPoints(bbox_contour)
            w = np.linalg.norm(bbox[0] - bbox[1])
            h = np.linalg.norm(bbox[1] - bbox[2])
            z3d = np.median(self.data.depth_image[box_candidate]) * self.data.depth_scale
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

    def visualize_detected_boxes(self) -> None:
        """
        Visualize the detected boxes.

        Returns:
            None
        """
        if not self.detected_boxes:
            print("No boxes detected.")
            return

        # Visualize the bounding box and box_candidate
        color_image = self.data.color_image.copy()
        for box, bbox in self.detected_boxes:
            cv2.drawContours(color_image, [bbox], 0, (255, 0, 0), 2)
            plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
            plt.contour(box, colors="r")

        plt.axis("off")
        plt.show()

    def _filter_masks(self, masks: List[np.ndarray]) -> List[np.ndarray]:
        """
        Filter masks to remove contained masks.

        Args:
            masks (List[np.ndarray]): List of masks.

        Returns:
            List[np.ndarray]: List of filtered masks.
        """
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
