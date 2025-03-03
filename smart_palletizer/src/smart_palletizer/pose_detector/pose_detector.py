import os
import copy
import pickle
import numpy as np
from utils.data_loading import PalletizerData
from surface_patches import detect_planar_surfaces
from clean_pointcloud import filter_outliers
import open3d as o3d
from omegaconf import DictConfig


class PoseDetector:
    """
    A class to detect poses of objects in an RGB-D image by using ICP given a binary image object mask and a target object point cloud.

    Attributes:
        point_cloud (o3d.geometry.PointCloud): The point cloud created from the RGB-D image.
        detected_poses (List[np.ndarray]): The detected poses of the objects.
        data (PalletizerData): The loaded data for the palletizer.

    Methods:
        detect_poses(): Detect poses of the object in the RGB-D image.
        visualize_detected_poses(): Visualize the detected poses.
    """

    def __init__(self, object: str, config: DictConfig) -> None:
        """
        Initialize the PoseDetector with the given object and configuration.

        Args:
            object (str): The object to detect poses for.
            config (DictConfig): The configuration dictionary.
        """
        data = PalletizerData(config.palletizer_data, object)

        self.detected_poses = None
        self.data = data
        self.config = config
        self.point_cloud = self._create_point_cloud(data.width, data.height, data.intrinsics, data.rgbd_image)

    def _create_point_cloud(
        self,
        width: int,
        height: int,
        intrinsics: np.ndarray,
        rgbd_image: o3d.geometry.RGBDImage,
        visualize: bool = False,
    ) -> o3d.geometry.PointCloud:
        """
        Create a point cloud from the RGB-D image and camera intrinsics.

        Args:
            width (int): The width of the image.
            height (int): The height of the image.
            intrinsics (np.ndarray): The camera intrinsics.
            rgbd_image (o3d.geometry.RGBDImage): The RGB-D image.
            visualize (bool): Whether to visualize the point cloud.

        Returns:
            o3d.geometry.PointCloud: The created point cloud.
        """
        # Create camera intrinsic object
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=width,
            height=height,
            fx=intrinsics[0, 0],
            fy=intrinsics[1, 1],
            cx=intrinsics[0, 2],
            cy=intrinsics[1, 2],
        )

        # Create a point cloud from the RGBD image and camera intrinsics
        radius = self.config.pose_detector.radius
        max_nn = self.config.pose_detector.max_nn
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        pcd.orient_normals_towards_camera_location()

        # Flip the point cloud to align with the Open3D coordinate system
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # Visualize the point cloud
        if visualize:
            o3d.visualization.draw_geometries([pcd], point_show_normal=True)

        return pcd

    def detect_poses(self) -> None:
        """
        Detect poses of the object in the RGB-D image. Uses pre-computed image masks for single boxes

        Returns:
            None
        """
        detected_poses = []

        for binary_mask, _ in self.data.detected_boxes:
            detected_poses.append(self._detect_pose(binary_mask))

        with open(os.path.join(self.data.data_path, "detected_poses.pkl"), "wb") as f:
            pickle.dump(self.detected_poses, f)

        self.detected_poses = detected_poses

    def _detect_pose(self, binary_mask: np.ndarray, visualize: bool = False) -> np.ndarray:
        """
        Detect the pose of a box in the underlying RGB-D image using the given binary image mask and ICP.

        Args:
            binary_mask (np.ndarray): The binary mask of a box.
            visualize (bool): Whether to visualize the detected pose.

        Returns:
            np.ndarray: The transformation matrix of the detected pose.
        """
        # Step 1: Mask the RGBD image
        masked_color = np.asarray(self.data.rgbd_image.color).copy()
        masked_depth = np.asarray(self.data.rgbd_image.depth).copy()

        masked_color[~binary_mask] = 0
        masked_depth[~binary_mask] = 0

        masked_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(masked_color),
            o3d.geometry.Image(masked_depth),
            depth_scale=1.0,
            convert_rgb_to_intensity=False,
        )

        # Step 2: Create a point cloud from the masked RGBD image
        masked_pcd = self._create_point_cloud(self.data.width, self.data.height, self.data.intrinsics, masked_rgbd)
        masked_pcd = filter_outliers(masked_pcd, self.config.filter_outliers)

        planes, _ = detect_planar_surfaces(masked_pcd, self.config.plane_detection)
        # Merge all detected planes into a single point cloud
        masked_pcd = o3d.geometry.PointCloud()
        for plane in planes:
            masked_pcd += plane

        # Step 3: Fit the given box mesh to the point cloud
        # We'll use ICP (Iterative Closest Point) algorithm for this
        box_pcd = copy.deepcopy(self.data.box_pcd)
        initial_transformation = np.identity(4)  # Initial transformation estimate
        # Calculate the mean x, y, z coordinates from the masked point cloud
        initial_transformation[0:3, 3] = np.mean(np.asarray(masked_pcd.points), axis=0)

        max_correspondence_distance = self.config.pose_detector.max_correspondence_distance
        max_iteration = self.config.pose_detector.max_iteration
        icp_result = o3d.pipelines.registration.registration_icp(
            box_pcd,
            masked_pcd,
            max_correspondence_distance=max_correspondence_distance,
            init=initial_transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration),
        )

        # Apply the transformation to the mesh
        box_pcd.transform(icp_result.transformation)

        # Visualize the point clouds and coordinate frames
        if visualize:
            # Create coordinate frames for visualization
            masked_pcd_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            box_pcd_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            # Apply the same transformation to the box_pcd_frame
            box_pcd_frame.transform(icp_result.transformation)
            o3d.visualization.draw_geometries([masked_pcd, box_pcd, masked_pcd_frame, box_pcd_frame])

        return icp_result.transformation

    def visualize_detected_poses(self) -> None:
        """
        Visualize the detected poses.

        Returns:
            None
        """
        if not self.detected_poses:
            print("No poses detected.")
            return

        vis_objects = [self.point_cloud]
        for pose in self.detected_poses:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            frame.transform(pose)
            vis_objects.append(frame)

        o3d.visualization.draw_geometries(vis_objects)
