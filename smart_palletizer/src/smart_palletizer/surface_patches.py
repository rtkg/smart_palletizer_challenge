import open3d as o3d
import argparse
import os
import numpy as np


def detect_planar_surfaces(input_path, output_path, visualize=True):
    """
    Detect planar surfaces in a point cloud using RANSAC.

    Args:
        input_path (str): Path to the input point cloud file.
        output_path (str): Path to save the point cloud with detected planes.
        visualize (bool): Whether to visualize the point cloud with detected planes.
    """
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(input_path)

    # List to store detected planes
    planes = []
    plane_models = []
    # Parameters for plane segmentation
    distance_threshold = 0.01
    ransac_n = 3
    num_iterations = 1000

    while True:
        # Segment the largest plane in the point cloud
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations
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
        if len(pcd.points) < 500:
            break

    # Assign random colors to each detected plane
    for plane in planes:
        color = np.random.rand(3)
        plane.paint_uniform_color(color)

    if visualize:
        o3d.visualization.draw_geometries(planes)

    # Save the plane models to the output path
    np.save(output_path, plane_models)


def project_points_to_plane(points, plane_model):
    """
    Projects a set of 3D points onto a specified plane.

    Args:
        points (open3d.geometry.PointCloud): The input point cloud containing the 3D points to be projected.
        plane_model (list or array-like): The coefficients [a, b, c, d] of the plane equation ax + by + cz + d = 0.

    Returns:
        open3d.geometry.PointCloud: A new point cloud containing the projected 3D points onto the plane.
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

    parser = argparse.ArgumentParser(description="Detect planar surfaces in a point cloud and save the result.")
    parser.add_argument(
        "--input_path", type=str, nargs="?", default=default_input_path, help="Path to the input point cloud file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        nargs="?",
        default=default_output_path,
        help="Path to save the point cloud with detected planes.",
    )
    parser.add_argument("--visualize", action="store_false", help="Visualize the point cloud with detected planes.")
    args = parser.parse_args()

    detect_planar_surfaces(args.input_path, args.output_path, args.visualize)
