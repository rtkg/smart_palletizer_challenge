import os
import itertools
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline
import pickle
from utils.data_loading import PalletizerData
from utils.visualization import show_mask
from surface_patches import detect_planar_surfaces
from clean_pointcloud import filter_outliers
import open3d as o3d
from omegaconf import DictConfig
import copy


class PoseDetector:
    def __init__(self, object, config: DictConfig):
        data = PalletizerData(config, object)

        self.point_cloud = self._create_point_cloud(data.width, data.height, data.intrinsics, data.rgbd_image)
        self.detected_poses = None
        self.data = data

    def _create_point_cloud(self, width, height, intrinsics, rgbd_image, visualize=False):

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
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_towards_camera_location()
        # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Flip the point cloud to align with the Open3D coordinate system
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # Visualize the point cloud
        if visualize:
            o3d.visualization.draw_geometries([pcd], point_show_normal=True)

        return pcd

    def detect_poses(self):
        detected_poses = []

        for binary_mask, _ in self.data.detected_boxes:
            detected_poses.append(self._detect_pose(binary_mask))

        with open(os.path.join(self.data.data_path, "detected_poses.pkl"), "wb") as f:
            pickle.dump(self.detected_poses, f)

        self.detected_poses = detected_poses

    def _detect_pose(self, binary_mask, visualize=False):
        # Step 1: Mask the RGBD image
        color = np.asarray(self.data.rgbd_image.color)
        depth = np.asarray(self.data.rgbd_image.depth)
        masked_color = color.copy()
        masked_depth = depth.copy()

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
        # masked_pcd = filter_outliers(masked_pcd)

        planes, _ = detect_planar_surfaces(masked_pcd)
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

        icp_result = o3d.pipelines.registration.registration_icp(
            box_pcd,
            masked_pcd,
            max_correspondence_distance=0.05,
            init=initial_transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000),
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

    def visualize_detected_poses(self):
        if not self.detected_poses:
            print("No poses detected.")
            return

        vis_objects = [self.point_cloud]
        for pose in self.detected_poses:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            frame.transform(pose)
            vis_objects.append(frame)

        o3d.visualization.draw_geometries(vis_objects)
