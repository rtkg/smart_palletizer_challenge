import pickle
import json
import os
from typing import Tuple
import cv2
import open3d as o3d
import numpy as np


class PalletizerData:
    """
    Helper class to load and manage data for the palletizer.
    """

    def __init__(self, config: dict, object: str) -> None:
        """
        Initialize the PalletizerData with the given configuration and object.

        Args:
            config (dict): The configuration dictionary.
            object (str): The object to load data for (can be 'medium_box' or 'small_box').
        """
        self.depth_scale = config["depth_scale"]
        current_file_path = os.path.abspath(__file__)
        data_path = os.path.join(os.path.dirname(current_file_path), "../../../data/", object)

        color_image_path = os.path.join(data_path, "color_image.png")
        depth_image_path = os.path.join(data_path, "raw_depth.png")
        intrinsics_path = os.path.join(data_path, "intrinsics.json")
        extrinsics_path = os.path.join(data_path, "cam2root.json")
        box_mesh_path = os.path.join(data_path, object + "_mesh.ply")
        detected_boxes_path = os.path.join(data_path, "detected_boxes.pkl")
        if not os.path.exists(detected_boxes_path):
            raise FileNotFoundError(f"Pickle file not found: {detected_boxes_path}")
        with open(detected_boxes_path, "rb") as f:
            self.detected_boxes = pickle.load(f)

        self.color_image = load_image(color_image_path)
        self.depth_image = load_image(depth_image_path, is_depth_image=True)

        color_raw = o3d.geometry.Image(self.color_image)
        depth_raw = o3d.geometry.Image(self.depth_image.astype(np.float32) * self.depth_scale)
        self.rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, depth_scale=1, convert_rgb_to_intensity=False
        )

        self.width, self.height, self.intrinsics, self.extrinsics = load_camera_parameters(
            intrinsics_path, extrinsics_path
        )
        box_mesh = load_mesh(box_mesh_path)
        self.box_pcd = box_mesh.sample_points_uniformly(number_of_points=10000)
        self.box_pcd.scale(1 / 1000, center=(0, 0, 0))
        self.box_pcd.normals = o3d.utility.Vector3dVector(np.asarray(box_mesh.vertex_normals))

        self.data_path = data_path

        self.object = object

        if object == "small_box":
            self.box_dimensions = config["small_box_dimensions"]  # Use config value
        elif object == "medium_box":
            self.box_dimensions = config["medium_box_dimensions"]  # Use config value
        else:
            raise ValueError("Invalid object type. Supported objects are 'small_box' and 'medium_box'.")


def load_image(image_path: str, is_depth_image: bool = False) -> np.ndarray:
    """
    Load and display a depth or rgb image.

    Args:
        image_path (str): Path to the image file.
        is_depth_image (bool): Flag indicating if the image is a depth image.

    Returns:
        np.ndarray: Loaded image.

    Raises:
        FileNotFoundError: If the image file could not be loaded.
    """

    # Load image
    if is_depth_image:
        image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    else:
        image = cv2.imread(image_path)

    # Check if images were loaded successfully
    if image is None:
        raise FileNotFoundError(f"Image file not found or could not be loaded: {image_path}")

    return image


def load_camera_parameters(intrinsics_path: str, extrinsics_path: str) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    Load camera intrinsics and extrinsics from JSON files.

    Args:
        intrinsics_path (str): Path to the intrinsics JSON file.
        extrinsics_path (str): Path to the extrinsics JSON file.

    Returns:
        tuple: Loaded image resolution, intrinsics and extrinsics.

    Raises:
        FileNotFoundError: If any of the JSON files could not be loaded.
    """

    if not os.path.exists(intrinsics_path):
        raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_path}")
    if not os.path.exists(extrinsics_path):
        raise FileNotFoundError(f"Extrinsics file not found: {extrinsics_path}")

    with open(intrinsics_path, "r") as f:
        intrinsics = json.load(f)
        intrinsics_matrix = np.eye(3)
        intrinsics_matrix[0, 0] = intrinsics["fx"]
        intrinsics_matrix[0, 2] = intrinsics["cx"]
        intrinsics_matrix[1, 1] = intrinsics["fy"]
        intrinsics_matrix[1, 2] = intrinsics["cy"]

    with open(extrinsics_path, "r") as f:
        extrinsics = json.load(f)
        # convert dict to numpy array
        extrinsics_matrix = np.array(list(extrinsics.values())[0])

    return intrinsics["width"], intrinsics["height"], intrinsics_matrix, extrinsics_matrix


def load_point_cloud(pcd_path: str, visualize: bool = False) -> o3d.geometry.PointCloud:
    """
    Load a point cloud from a PLY file using Open3D.

    Args:
        pcd_path (str): Path to the PLY file.
        visualize (bool): Flag indicating if the point cloud should be visualized.

    Returns:
        o3d.geometry.PointCloud: Loaded point cloud.

    Raises:
        FileNotFoundError: If the PLY file could not be loaded.
    """
    if not os.path.exists(pcd_path):
        raise FileNotFoundError(f"PLY file not found: {pcd_path}")

    # Load the PLY file
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_towards_camera_location()

    # Flip the point cloud to align with the Open3D coordinate system
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # Visualize the point cloud
    if visualize:
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    return pcd


def load_mesh(mesh_path: str, visualize: bool = False) -> o3d.geometry.TriangleMesh:
    """
    Load a PLY file using Open3D.

    Args:
        mesh_path (str): Path to the PLY file.
        visualize (bool): Flag indicating if the mesh should be visualized.

    Returns:
        o3d.geometry.TriangleMesh: Loaded mesh.

    Raises:
        FileNotFoundError: If the PLY file could not be loaded.
    """
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"mesh file not found: {mesh_path}")

    # Load the PLY file
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    if visualize:
        o3d.visualization.draw_geometries([mesh])

    return mesh
