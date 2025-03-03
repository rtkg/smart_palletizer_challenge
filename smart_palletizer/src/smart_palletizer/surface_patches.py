"""
Script parses command-line arguments and calls the patch_detection function. Used to detect planar 
surfaces in point clouds representing boxes.

Example usage:
    python surface_patches.py -i /path/to/input.ply -o /path/to/output.npy -v

Command-line arguments:
    --input_path, -i (str): Path to the input point cloud file. Defaults to a sample file path.
    --output_path, -o (str): Path to save the point cloud with detected planes. Defaults to a sample output path.
    --visualize, -v (bool): Flag to visualize the point cloud with detected planes. Defaults to False.
"""

import os
from typing import List, Tuple
import argparse
import pickle
import open3d as o3d
import numpy as np
from omegaconf import OmegaConf, DictConfig


def patch_detection(input_path: str, output_path: str, config: DictConfig, visualize: bool = False) -> None:
    """
    Detect planar surfaces in a point cloud using RANSAC.

    Args:
        input_path (str): Path to the input point cloud file.
        output_path (str): Path to save the point cloud with detected planes.
        config (DictConfig): Configuration parameters for patch detection.
        visualize (bool): Whether to visualize the point cloud with detected planes.

    Returns:
        None
    """
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(input_path)
    if pcd.is_empty():
        raise ValueError(f"The point cloud is empty. Please check the input path: {input_path}")

    planes, plane_models = detect_planar_surfaces(pcd, config.plane_detection, visualize=visualize)

    # Save the plane models to the output path
    with open(output_path, "wb") as f:
        pickle.dump(([np.asarray(plane.points) for plane in planes], plane_models), f)


def detect_planar_surfaces(
    pcd: o3d.geometry.PointCloud, config: DictConfig, visualize: bool = False
) -> Tuple[List[o3d.geometry.PointCloud], List[np.ndarray]]:
    """
    Detect planar surfaces in a point cloud using RANSAC.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        config (DictConfig): Configuration parameters for plane detection.
        visualize (bool): Whether to visualize the point cloud with detected planes.

    Returns:
        Tuple[List[o3d.geometry.PointCloud], List[np.ndarray]]: A tuple containing the list of detected
        planes and their models.
    """
    # List to store detected planes
    planes = []
    plane_models = []

    while True:
        # Segment the largest plane in the point cloud using RANSAC
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=config.distance_threshold, ransac_n=config.ransac_n, num_iterations=config.num_iterations
        )
        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud = project_points_to_plane(inlier_cloud, plane_model)

        outlier_cloud = pcd.select_by_index(inliers, invert=True)

        # Add the detected plane to the list
        planes.append(inlier_cloud)
        plane_models.append(plane_model)
        # Update the point cloud to remove the detected plane
        pcd = outlier_cloud

        # Stop if the remaining point cloud is too small
        if len(pcd.points) < config.min_points:
            break

    # Assign random colors to each detected plane
    for plane in planes:
        color = np.random.rand(3)
        plane.paint_uniform_color(color)
        plane.estimate_normals()
        plane.orient_normals_towards_camera_location()

    if visualize:
        o3d.visualization.draw_geometries(planes)

    return planes, plane_models


def project_points_to_plane(points: o3d.geometry.PointCloud, plane_model: List[float]) -> o3d.geometry.PointCloud:
    """
    Projects a set of 3D points onto a specified plane.

    Args:
        points (o3d.geometry.PointCloud): The input point cloud containing the 3D points to be projected.
        plane_model (List[float]): The coefficients [a, b, c, d] of the plane equation ax + by + cz + d = 0.

    Returns:
        o3d.geometry.PointCloud: A new point cloud containing the projected 3D points onto the plane.
    """
    [a, b, c, d] = plane_model
    plane_normal = np.array([a, b, c])
    plane_normal /= np.linalg.norm(plane_normal)

    points_array = np.asarray(points.points)
    distances = (np.dot(points_array, plane_normal) + d) / np.linalg.norm(plane_normal)
    projected_points = points_array - np.outer(distances, plane_normal)

    projected_cloud = o3d.geometry.PointCloud()
    projected_cloud.points = o3d.utility.Vector3dVector(projected_points)

    return projected_cloud


if __name__ == "__main__":

    current_file_path = os.path.abspath(__file__)
    default_input_path = os.path.join(
        os.path.dirname(current_file_path), "../../data/medium_box/medium_box_0_cleaned.ply"
    )
    default_output_path = os.path.join(
        os.path.dirname(current_file_path), "../../data/medium_box/medium_box_0_planes.npy"
    )
    config_path = os.path.join(os.path.dirname(current_file_path), "../../config/surface_patches.yaml")

    parser = argparse.ArgumentParser(description="Detect planar surfaces in a point cloud and save the result.")
    parser.add_argument(
        "--input_path",
        "-i",
        type=str,
        nargs="?",
        default=default_input_path,
        help="Path to the input point cloud file.",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        nargs="?",
        default=default_output_path,
        help="Path to save the point cloud with detected planes.",
    )
    parser.add_argument(
        "--visualize", "-v", action="store_true", help="Visualize the point cloud with detected planes."
    )
    args = parser.parse_args()

    # Load configuration
    config = OmegaConf.load(config_path)

    patch_detection(args.input_path, args.output_path, config, args.visualize)
