import os
from mmap import mmap

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

def mapcount(f):
    buf = mmap(f.fileno(), 0)
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    return lines

def load_colmap_points3d(path, i=0):
    points = []
    colors = []

    is_bin = path.endswith('.bin')
    if is_bin:
        l = len("points3D.bin")
        output_path = path.replace(f"/{i}/", f"/{i}_txt/")[:-l]
        os.system('mkdir -p ' + output_path)
        os.system(f"colmap model_converter --input_path {path[:-l]} --output_path {output_path} --output_type TXT")
                
        path = output_path + "/points3D.txt"

    with open(path, 'r+') as f:
        for line in tqdm(f, desc="Loading points3D.txt", total=mapcount(f)):
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 7:  # Skip invalid lines (ID, X, Y, Z, R, G, B, ...)
                continue
            points.append([float(parts[1]), float(parts[2]), float(parts[3])])  # XYZ
            colors.append([float(parts[4])/255, float(parts[5])/255, float(parts[6])/255])  # RGB
    return np.array(points), np.array(colors)

# Load sparse reconstruction
path = "points3D.txt"
points, colors = np.array([]), np.array([])
for i in range(10)  :
    p, c = load_colmap_points3d(f"data/middle_colmap/sparse/{i}_txt/{path}", i=i)
    points = np.vstack((points, p)) if points.size else p
    colors = np.vstack((colors, c)) if colors.size else c

def downsample_points(points, colors, ratio=0.5):
    num_points = len(points)
    indices = np.random.choice(num_points, int(num_points * ratio), replace=False)
    return points[indices], colors[indices]

# Downsample to 10% of points
downsampled_points, downsampled_colors = downsample_points(points, colors, ratio=0.99)

def plot_sparse_points(points, colors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Sparse Point Cloud")
    plt.show()

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(downsampled_points)
pcd.colors = o3d.utility.Vector3dVector(downsampled_colors)
o3d.visualization.draw_geometries([pcd], window_name=f"Sparse Cloud {path}")