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
import open3d as o3d
from omegaconf import DictConfig


class PoseDetector:
    def __init__(self, object, config: DictConfig):
        self.data = PalletizerData(config, object)

        self.point_cloud = self.create_point_cloud(self.data)
        self.detected_poses = None

    def create_point_cloud(self, data):

        # Create camera intrinsic object
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=data.width,
            height=data.height,
            fx=data.intrinsics[0, 0],
            fy=data.intrinsics[1, 1],
            cx=data.intrinsics[0, 2],
            cy=data.intrinsics[1, 2],
        )

        # Create a point cloud from the RGBD image and camera intrinsics
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(data.rgbd_image, intrinsic)

        # Flip the point cloud to align with the Open3D coordinate system
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # Visualize the point cloud
        # o3d.visualization.draw_geometries([pcd])

        return pcd

    def detect_poses(self):
        # Step 1: Mask the RGBD image
        color = np.asarray(self.data.rgbd_image.color)
        depth = np.asarray(self.data.rgbd_image.depth)
        binary_mask = self.data.detected_boxes[0][0]
        masked_color = color.copy()
        masked_depth = depth.copy()

        masked_color[~binary_mask] = 0
        masked_depth[~binary_mask] = 0

        masked_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(masked_color), o3d.geometry.Image(masked_depth), convert_rgb_to_intensity=False
        )

        # Step 2: Create a point cloud from the masked RGBD image
        intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(masked_rgbd, intrinsic)

        # Step 3: Fit the given mesh to the point cloud
        # We'll use ICP (Iterative Closest Point) algorithm for this
        icp_result = o3d.pipelines.registration.registration_icp(
            mesh,
            pcd,
            max_correspondence_distance=0.05,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
        )

        # Apply the transformation to the mesh
        mesh_aligned = mesh.transform(icp_result.transformation)

        # Visualize the result
        o3d.visualization.draw_geometries([pcd, mesh_aligned])

    def detect_poses_(self):
        detected_poses = []

        # match data.box_mesh to all data.detected_boxes masked in self.point_cloud
        for box, bbox in self.data.detected_boxes:
            # Create a mask for the detected box
            mask = np.zeros((self.data.height, self.data.width), dtype=np.uint8)
            cv2.drawContours(mask, [bbox], -1, 255, thickness=cv2.FILLED)

            # Apply the mask to the point cloud
            masked_points = np.asarray(self.point_cloud.points)[mask.flatten() == 255]
            masked_colors = np.asarray(self.point_cloud.colors)[mask.flatten() == 255]

            # Create a new point cloud with the masked points
            masked_pcd = o3d.geometry.PointCloud()
            masked_pcd.points = o3d.utility.Vector3dVector(masked_points)
            masked_pcd.colors = o3d.utility.Vector3dVector(masked_colors)

            # Match the box mesh to the masked point cloud using ICP
            threshold = 0.02  # You can adjust this threshold
            init_transformation = np.identity(4)  # Initial transformation

            result = o3d.pipelines.registration.registration_icp(
                self.data.box_mesh,
                masked_pcd,
                threshold,
                init_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )
            transformation = result.transformation

            # Apply the transformation to the box mesh
            transformed_box_mesh = self.data.box_mesh.transform(transformation)

            # Append the transformed box mesh and bbox to detected_poses
            detected_poses.append((transformed_box_mesh, bbox))

        self.detected_poses = detected_poses

        with open(os.path.join(self.data.data_path, "detected_poses.pkl"), "wb") as f:
            pickle.dump(self.detected_poses, f)

    def visualize_detected_poses(self):
        if not self.detected_poses:
            print("No poses detected.")
            return

        # Visualize the bounding box and box_candidate
        color_image = self.color_image.copy()
        for box, bbox in self.detected_poses:
            cv2.drawContours(color_image, [bbox], 0, (255, 0, 0), 2)
            plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
            plt.contour(box, colors="r")

        plt.axis("off")
        plt.show()
