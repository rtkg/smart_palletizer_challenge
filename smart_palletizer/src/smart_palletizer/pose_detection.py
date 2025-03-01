from utils.data_loading import load_mesh, load_image, visualize_image, load_camera_parameters
import os
import argparse

import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline
from scipy.ndimage import label
from sklearn.cluster import KMeans
import open3d as o3d

# Determine the package root folder path
# package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Construct the paths to the meshes, color and depth images
# color_image_path = os.path.join(package_root, "data/medium_box/color_image.png")
# depth_image_path = os.path.join(package_root, "data/medium_box/raw_depth.png")
# box_mask_0_path = os.path.join(package_root, "data/medium_box/medium_box_mask_0.png")
# box_mask_1_path = os.path.join(package_root, "data/medium_box/medium_box_mask_1.png")
# box_mesh_path = os.path.join(package_root, "data/medium_box/medium_box_mesh.ply")

# Load the meshes, color and depth images from the medium_box folder
# color_image = load_image(color_image_path)
# depth_image = load_image(depth_image_path, is_depth_image=True)
# box_mask_0_image = load_image(box_mask_0_path)
# box_mask_1_image = load_image(box_mask_1_path)
# box_mesh = load_mesh(box_mesh_path)
# o3d.visualization.draw_geometries(box_mesh)

# visualize_image(box_mask_1_image, is_depth_image=True)
# visualize_image(depth_image, is_depth_image=True)

# intrinsics_path = os.path.join(package_root, "data/medium_box/intrinsics.json")
# extrinsics_path = os.path.join(package_root, "data/medium_box/cam2root.json")

# width, height, intrinsics, extrinsics = load_camera_parameters(intrinsics_path, extrinsics_path)


def detect_boxes(color_image_path, depth_image_path, visualize=False):

    color_image = load_image(color_image_path)
    depth_image = load_image(depth_image_path, is_depth_image=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device=device)

    color_image_pil = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    masks = generator(color_image_pil, points_per_batch=128, pred_iou_thresh=0.95)

    masks = filter_masks(masks["masks"], 500)

    # min_depth = np.min(depth_image)
    # max_depth = np.max(depth_image)
    # depth_bins = np.linspace(min_depth, max_depth, 40)
    # merged_masks = []
    # for i in range(len(depth_bins) - 1):
    #    depth_bin = (depth_bins[i], depth_bins[i + 1])
    #    merged_mask = merge_masks_by_depth_range(masks, depth_image, depth_bin[0], depth_bin[1])
    #    if np.any(merged_mask):
    #        merged_masks.append((merged_mask, depth_bin))

    # Create an Open3D point cloud from the color and depth images
    color_raw = o3d.geometry.Image(color_image)
    depth_raw = o3d.geometry.Image(depth_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, convert_rgb_to_intensity=False
    )

    # Create a point cloud from the RGBD image and camera intrinsics
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    )

    # Flip the point cloud to align with the Open3D coordinate system
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

    merged_masks = masks

    if visualize:
        ax = plt.gca()
        for mask in merged_masks:
            show_mask(mask, ax=ax, random_color=True)

        plt.axis("off")
        plt.show()


def filter_masks(masks):
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

        # filter by shape
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        rect_area = w * h
        contour_area = cv2.contourArea(contours[0])
        area_difference = abs(rect_area - contour_area) / rect_area
        if area_difference > 0.2:
            continue

        if not is_contained:
            filtered_masks.append(mask)

    # add a filter that checks whether boxes are rectangular

    return filtered_masks


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def merge_masks_by_depth_range(masks, depth_image, depth_min, depth_max):
    # Initialize the merged mask
    merged_mask = np.zeros_like(masks[0], dtype=bool)

    for mask in masks:
        masked_depth_image = depth_image[mask]

        if np.median(masked_depth_image) >= depth_min and np.median(masked_depth_image) < depth_max:
            merged_mask |= mask

    return merged_mask


if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    default_input_path = os.path.join(os.path.dirname(current_file_path), "../../data/medium_box/color_image.png")
    default_output_path = os.path.join(os.path.dirname(current_file_path), "../../data/medium_box/raw_depth.png")

    parser = argparse.ArgumentParser(description="Detect boxes based on and RGB-D image ")
    parser.add_argument(
        "--color_image_path",
        type=str,
        nargs="?",
        default=default_input_path,
        help="Path to the input point cloud file.",
    )
    parser.add_argument(
        "--depth_image_path",
        type=str,
        nargs="?",
        default=default_output_path,
        help="Path to save the point cloud with detected planes.",
    )
    parser.add_argument("--visualize", action="store_false", help="Visualize the point cloud with detected planes.")
    args = parser.parse_args()

    detect_boxes(args.color_image_path, args.depth_image_path, args.visualize)
