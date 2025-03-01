import numpy as np
import cv2


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


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
