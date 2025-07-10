import os
import shutil

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


def keep_middle_points(point3d_dir, threshold=0.1):
    with open(point3d_dir, "r") as f:
        lines = f.readlines()
    out_path = point3d_dir.replace("points3D.txt", "points3D_middle.txt")
    middle = (0, 0, 0)
    minimum = (np.inf, np.inf, np.inf)
    maximum = (-np.inf, -np.inf, -np.inf)
    for l in tqdm(lines):
        if l.startswith("#"):
            continue
        parts = l.strip().split()
        if len(parts) < 7:
            continue
        x, y, z = map(float, parts[1:4])
        minimum = (min(minimum[0], x), min(minimum[1], y), min(minimum[2], z))
        maximum = (max(maximum[0], x), max(maximum[1], y), max(maximum[2], z))
    middle = ((minimum[0] + maximum[0]) / 2, (minimum[1] + maximum[1]) / 2, (minimum[2] + maximum[2]) / 2)

    total = 0
    for l in tqdm(lines):
        if l.startswith("#"):
            continue
        parts = l.strip().split()
        if len(parts) < 7:
            continue
        x, y, z = map(float, parts[1:4])
        if abs(x - middle[0])/maximum[0] < threshold and abs(y - middle[1])/maximum[1] < threshold:
            with open(out_path, "a") as out:
                out.write(l)
                total += 1
    print(f"Total points kept: {total}, Middle point: {middle}, Min: {minimum}, Max: {maximum}")


def add_symlink(folder, output_folder, dir):
    """
    Create a symlink for the image directory.
    Args:
        folder (str): Base folder where images are stored.
        dir (str): Directory name to create a symlink for.
    """
    import os
    import pathlib

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    target = os.path.join(folder, dir)
    link_name = os.path.join(output_folder, dir)
    link_subfolder = link_name.rsplit("/", maxsplit=1)[0]

    # Adding link subdirectory if it doesn't exist
    if not os.path.exists(link_subfolder):
        os.makedirs(link_subfolder)
    # Remove existing symlink or directory if it exists
    if os.path.islink(link_name) or os.path.exists(link_name):
        os.remove(link_name)

    # Create the symlink
    shutil.copy(target, link_name)



def keep_middle_images(image_dir_txt, image_dir, output_dir, threshold=0.1, symlink = False):
    with open(image_dir_txt, "r") as f:
        lines = f.readlines()
    out_path = image_dir_txt.replace("images.txt", "images_middle.txt")

    coords = np.array([])  # Initialize an empty array for coordinates
    for l in tqdm(lines):
        if l.startswith("#"):
            continue
        parts = l.strip().split()
        if len(parts) < 7:
            continue
        tx, ty, tz = map(float, parts[5:8])
        coords = np.append(coords, [tx, ty, tz]) if coords.size else np.array([tx, ty, tz])

    coords = coords.reshape(-1, 3)
    middle = np.mean(coords, axis=0)
    minimum = np.min(coords, axis=0)
    maximum = np.max(coords, axis=0)

    distances = np.linalg.norm(coords - middle, axis=1)
    threshold_distance = np.percentile(distances, threshold * 100)
    total = 0
    new_lines = []
    images = []
    for l in tqdm(lines):
            if l.startswith("#"):
                new_lines.append(l)
                continue
            parts = l.strip().split()
            if len(parts) < 8 :
                continue
            tx, ty, tz = map(float, parts[5:8])
            if np.linalg.norm([tx - middle[0], ty - middle[1], tz - middle[2]]) < threshold_distance:
                new_lines.append(l)
                new_lines.append("\n")
                total += 1
                dir = parts[-1]
                if symlink:
                    add_symlink(image_dir, output_dir, dir)
                else:
                    os.makedirs(os.path.join(output_dir, dir.split("/")[0]), exist_ok=True)
                    shutil.copy(os.path.join(image_dir, dir), os.path.join(output_dir, dir))
                images.append(dir)
    # Write the new lines to the output file

    with open(out_path, "w") as out:
        out.writelines(new_lines)
    print(f"Total images kept: {total}, Max distance: {threshold_distance}")
def subsample_points(input_file, output_file, min_distance=0.5):
    """
    Remove dense 3D points by enforcing a minimum distance between points.

    Args:
        input_file (str): Path to input file (COLMAP format or similar).
        output_file (str): Path to output file.
        min_distance (float): Minimum allowed distance between points (in meters).
    """
    # Step 1: Load points (assumes COLMAP format: POINT3D_ID, X, Y, Z, R, G, B, ...)
    points = []
    metadata = []  # Store other columns (e.g., RGB, track info)
    with open(input_file, 'r') as f:
        for line in tqdm(f):
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            x, y, z = map(float, parts[1:4])
            points.append([x, y, z])
            metadata.append(parts)  # Save full line for later

    points = np.array(points)
    print(f"Loaded {len(points)} points.")



    # Build a global KDTree for all points (read-only, shared across workers)
    global_tree = cKDTree(points)

    neighbors = global_tree.query_ball_tree(global_tree, min_distance)
    print(f"Found {len(neighbors)} points.")
    keep_mask = np.ones(len(points), dtype=bool)

    for i in tqdm(range(len(points))):
        if not keep_mask[i]:
            continue

        # Get all neighbors (including cross-chunk)
        neighbor_indices = list(neighbors[i])

        # Only mark subsequent points as False to avoid race conditions
        for j in neighbor_indices:
            if j > i:  # Important: only mark points after current
                keep_mask[j] = False


    # Step 3: Save subsampled points
    with open(output_file, 'w') as f_out:
        for idx in np.where(keep_mask)[0]:
            f_out.write(' '.join(metadata[idx]) + '\n')
        # write unkept points for verification

    print(f"Final kept points: {keep_mask.sum()}/{len(points)}")
