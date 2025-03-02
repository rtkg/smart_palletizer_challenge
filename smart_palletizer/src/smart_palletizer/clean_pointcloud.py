import argparse
import os
import open3d as o3d


def clean_point_cloud(input_path: str, output_path: str, visualize: bool = False) -> None:
    """
    Clean a point cloud by removing statistical and radius outliers.

    Args:
        input_path (str): Path to the input point cloud file.
        output_path (str): Path to save the cleaned point cloud file.
        visualize (bool): Whether to visualize the cleaned point cloud before saving.

    Returns:
        None
    """
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(input_path)

    cleaned_pcd = filter_outliers(pcd, visualize=visualize)

    # Save the cleaned point cloud
    o3d.io.write_point_cloud(output_path, cleaned_pcd)
    print(f"Cleaned point cloud saved to {output_path}")


def filter_outliers(pcd: o3d.geometry.PointCloud, visualize: bool = False) -> o3d.geometry.PointCloud:
    """
    Filter outliers from a point cloud using statistical and radius outlier removal.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        visualize (bool): Whether to visualize the cleaned point cloud.

    Returns:
        o3d.geometry.PointCloud: The cleaned point cloud.
    """
    # Remove statistical outliers
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=0.4)
    cleaned_pcd = pcd.select_by_index(ind)

    # Remove radius outliers
    _, ind = cleaned_pcd.remove_radius_outlier(nb_points=150, radius=0.1)
    cleaned_pcd = cleaned_pcd.select_by_index(ind)

    if visualize:
        o3d.visualization.draw_geometries([cleaned_pcd])

    return cleaned_pcd


if __name__ == "__main__":
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)
    default_file_path = os.path.join(os.path.dirname(current_file_path), "../../data/medium_box/medium_box_0_raw.ply")
    default_load_path = os.path.join(
        os.path.dirname(current_file_path), "../../data/medium_box/medium_box_0_cleaned.ply"
    )

    parser = argparse.ArgumentParser(description="Clean a point cloud and save the result.")
    parser.add_argument(
        "--input_path", type=str, nargs="?", default=default_file_path, help="Path to the input point cloud file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        nargs="?",
        default=default_load_path,
        help="Path to save the cleaned point cloud file.",
    )
    parser.add_argument("--visualize", action="store_false", help="Visualize the point cloud before saving.")
    args = parser.parse_args()

    clean_point_cloud(args.input_path, args.output_path, args.visualize)
