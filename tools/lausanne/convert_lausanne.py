import glob
import logging
import os
from collections import defaultdict

import cv2
import numpy as np
import trimesh
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.spatial import KDTree, cKDTree
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from tools.lausanne.preprocess.adjust_coordinates import normalize_coordinates



import concurrent.futures
from functools import partial

from tools.lausanne.preprocess.create_camera_image import parse_xml_to_colmap
from tools.lausanne.preprocess.lower_resolution import keep_middle_images

def temp(image_path):
    """
    Temporary function to process images.
    Args:
        image_path (str): Path to the image.
        output_dir (str): Directory to save the processed image.
    """
    with open(image_path, 'r') as f:
        lines = f.readlines()
    out_path = image_path.replace("images.txt", "images_temp.txt")

    total = 0
    write = []
    for l in tqdm(lines):
        if l.startswith("#"):
            write.append(l)
            continue

        parts = l.strip().split()
        if len(parts) < 7:
            write.append(l)
            continue
        image_path = parts[-1]
        image_path = image_path.split("/")[-1]
        parts[-1] = image_path
        write.append(" ".join(parts) + "\n")
        total += 1

    print(f"Total images processed: {total}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Lausanne dataset paths.")
    parser.add_argument("--xml_path", type=str, default="data/lausanne/cameras.xml", help="Path to the input XML file")
    parser.add_argument("--output_dir", type=str, default="data/lausanne/", help="Directory to save outputs")
    parser.add_argument("--image_dir", type=str, default="data/lausanne/images",
                        help="Directory containing images")
    args = parser.parse_args()

    xml_path = args.xml_path
    output_dir = args.output_dir
    image_dir = args.image_dir

    #color_point_cloud("data/lausanne/OBJs/Lausanne_OBJs_LODs", point_3d_dir.replace("points3D.txt", "points3D_with_colors.txt"))


    #keep_middle(point_3d_dir, threshold=0.1)
    #min_distance = 0.1
    #subsample_points(point_3d_dir.replace("points3D.txt", "points3D_middle.txt"), point_3d_dir.replace("points3D.txt", f"points3D_pruned.txt"), min_distance=min_distance)

    #offset = np.array([2537838.153451856, 1152387.8597814455, 514.7779249203819])
    #add_offset_to_3d_points(point_3d_dir, offset)

    #temp(output_dir + "/images.txt")

    #os.makedirs(output_dir, exist_ok=True)
    track_dict = parse_xml_to_colmap(xml_path, output_dir, image_dir)
    keep_middle_images(os.path.join(output_dir, "image.txt"), image_dir, output_dir, threshold=0.1)
    #change_point3d(point_3d_dir, track_dict)

    #normalize_coordinates(image_dir, output_dir, point_3d_dir)
