import cv2
import open3d as o3d
import numpy as np
import json
import os


def load_image(image_path, is_depth_image: bool = False) -> np.ndarray:
    """
    Load and display a depth or rgb image

    Args:
        image_path (str): Path to the  image file.

    Returns:
        np.ndarray: Loaded image
    Raises:
        FileNotFoundError: If the image file could not be loaded.
    """

    # Load image
    if is_depth_image:
        image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
        # Remove noise using a median filter
        image = cv2.medianBlur(image, 5)
    else:
        image = cv2.imread(image_path)

    # Check if images were loaded successfully
    if image is None:
        raise FileNotFoundError(f"Image file not found or could not be loaded: {image_path}")

    return image


def visualize_image(image: np.ndarray, is_depth_image: bool = False) -> None:
    """
    Visualizes an image using OpenCV.

    Parameters:
    image (np.ndarray): The image to be visualized. It can be a regular image or a depth image.
    is_depth_image (bool): A flag indicating whether the image is a depth image. Default is False.

    Returns:
    None
    """

    if is_depth_image:
        # Apply a color map to the depth image for better visualization
        depth_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        depth_image = depth_image.astype(np.uint8)
        cv2.imshow("Depth Image", depth_image)
    else:
        cv2.imshow("Image", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_camera_parameters(intrinsics_path, extrinsics_path):
    """
    Load camera intrinsics and extrinsics from JSON files.

    Args:
        intrinsics_path (str): Path to the intrinsics JSON file.
        extrinsics_path (str): Path to the extrinsics JSON file.

    Returns:
        Tuple[int, int, np.ndarray, np.ndarray]: Loaded image resolution, intrinsics and extrinsics.
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


def load_mesh(mesh_path: str) -> None:
    """
    Load and visualize a PLY file using Open3D.

    Args:
        ply_file_path (str): Path to the PLY file.

    Raises:
        FileNotFoundError: If the PLY file could not be loaded.
    """
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"mesh file not found: {mesh_path}")

    # Load the PLY file
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])

    return mesh
