import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def qvec2rotmat(qvec):
    """Convert Hamilton quaternion to rotation matrix"""
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]
    ])


def project_point_to_image(point3D, qvec, tvec, camera_params):
    """
    Project 3D point to 2D image coordinates
    point3D: [X, Y, Z]
    qvec: [qw, qx, qy, qz]
    tvec: [tx, ty, tz]
    camera_params: [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6]
    """
    # Convert to numpy arrays
    point3D = np.array(point3D)
    tvec = np.array(tvec)

    # Convert quaternion to rotation matrix
    R_mat = qvec2rotmat(qvec)

    # Transform to camera coordinates
    P_cam = R_mat @ (point3D - tvec)

    # Normalize
    x = P_cam[0] / P_cam[2]
    y = P_cam[1] / P_cam[2]

    # Apply distortion (FULL_OPENCV model)
    r2 = x ** 2 + y ** 2
    radial = 1 + camera_params[4] * r2 + camera_params[5] * r2 ** 2 + camera_params[8] * r2 ** 3
    x_dist = x * radial + 2 * camera_params[6] * x * y + camera_params[7] * (r2 + 2 * x ** 2)
    y_dist = y * radial + camera_params[6] * (r2 + 2 * y ** 2) + 2 * camera_params[7] * x * y

    # Project to pixel coordinates
    u = camera_params[0] * x_dist + camera_params[2]
    v = camera_params[1] * y_dist + camera_params[3]

    return (u, v)


def visualize_projected_points(image_path, projected_points, output_path="projection_verification.jpg"):
    """
    Visualize projected 2D points on an image.

    Args:
        image_path (str): Path to the image file.
        projected_points (list): List of (u, v) coordinates.
        output_path (str): Path to save the output image.
    """
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Plot each point
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    for u, v in projected_points:
        plt.scatter(u, v, s=10, c='red', marker='o')  # Plot points in red

    plt.title("Projected 3D Points on Image")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Verification image saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Update these paths to match your dataset
    DATASET_PATH = "../../data/mini/dense"
    IMAGE_PATH = f"{DATASET_PATH}/images"
    SPARSE_PATH = f"{DATASET_PATH}/sparse"
    POINT_3D_PATH = f"{SPARSE_PATH}/images.txt"

    uv = []

    with open(POINT_3D_PATH, 'r+') as f:
        for i, line in tqdm(enumerate(f), desc="Loading points3D.txt"):
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if i % 2 == 0:
                continue
            for j in range(len(parts) // 3):
                u = float(parts[j * 3])
                v = float(parts[j * 3 + 1])

                uv.append((u, v))
            break
qxx
    result_img = visualize_projected_points(IMAGE_PATH, uv)