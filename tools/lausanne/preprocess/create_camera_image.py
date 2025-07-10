import glob
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import pyrender

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from PIL import Image



def rotation_matrix_to_quaternion(rotation_matrix):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    rotation = Rotation.from_matrix(rotation_matrix)
    quat = rotation.as_quat()  # Returns [x, y, z, w]
    return quat[3], quat[0], quat[1], quat[2]  # Reorder to (w, x, y, z)

def get_meshes(OBJ_PATH):
    obj_files = glob.glob(f"{OBJ_PATH}/*.obj")
    merged_mesh = trimesh.Trimesh()
    meshes = []
    if os.path.join(OBJ_PATH, "merged_mesh.obj") in obj_files:
        print("Merged mesh already exists, skipping merging step.")
        merged_mesh = trimesh.load(os.path.join(OBJ_PATH, "merged_mesh.obj"), process=False)
        return merged_mesh, merged_mesh.vertices
    for obj_file in tqdm(obj_files):
        # Load the scene
        scene = trimesh.load(obj_file, process=False)
        if isinstance(scene, trimesh.Scene):
            # Get all meshes from the scene
            meshes_scene = list(scene.geometry.values())
            if not meshes_scene:
                print(f"Skipping {obj_file}: no meshes found")
                continue
            meshes.extend(meshes_scene)
        else:
            meshes.append(scene)

    # Merge all meshes into one
    #save the merged mesh

    vertices = []
    faces = []
    for mesh in tqdm(meshes, desc="Merging meshes"):
        vertices.append(mesh.vertices)
        faces.append(mesh.faces)  # Offset faces by current vertex count

    offset = np.array([2537838.153451856, 1152387.8597814455, 514.7779249203819])
    minx = np.min(np.vstack(vertices)[:, 0])
    miny = np.min(np.vstack(vertices)[:, 1])
    minz = np.min(np.vstack(vertices)[:, 2])

    vertices = [v - np.array([minx, miny, minz]) + offset for v in vertices]  # Adjust vertices by the offset

    vertices = np.vstack(vertices)
    faces = np.vstack(faces)
    merged_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    merged_mesh.export(os.path.join(OBJ_PATH, "merged_mesh.obj"))

    print("Meshes merged")

    points_3d = vertices

    return merged_mesh, points_3d



def parse_xml_to_colmap(xml_path, output_dir, img_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    photo_id = 0
    #merged_mesh, point_3d = get_meshes(obj_path)

    with open(os.path.join(output_dir, "cameras.txt"), "w") as f_cam, \
         open(os.path.join(output_dir, "images.txt"), "w") as f_img:

        f_cam.write("# Camera list with one line of data per camera:\n")
        f_cam.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f_img.write("# Image list with two lines of data per image:\n")
        f_img.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")

        camera_id = 1  # Start counting camera IDs
        camera_params = {}  # Track unique cameras to avoid duplicates

        track_dict = defaultdict(list)
        skip = 0



        for photogroup in tqdm(root.findall(".//Photogroup")):
            # Extract intrinsics
            width = int(photogroup.find("ImageDimensions/Width").text)
            height = int(photogroup.find("ImageDimensions/Height").text)
            focal_length = float(photogroup.find("FocalLength").text)

            # convert focal to pixels
            sensor_width = float(photogroup.find("SensorSize").text)
            focal_length_pixels = focal_length * width / sensor_width

            px = float(photogroup.find("PrincipalPoint/x").text)
            py = float(photogroup.find("PrincipalPoint/y").text)
            k1 = float(photogroup.find("Distortion/K1").text) if photogroup.find("Distortion/K1") is not None else 0.0
            k2 = float(photogroup.find("Distortion/K2").text) if photogroup.find("Distortion/K2") is not None else 0.0
            k3 = float(photogroup.find("Distortion/K3").text) if photogroup.find("Distortion/K3") is not None else 0.0
            if photogroup.find("Distortion/K4") is not None:
                k4 = float(photogroup.find("Distortion/K4").text)
            p1 = float(photogroup.find("Distortion/P1").text) if photogroup.find("Distortion/P1") is not None else 0.0
            p2 = float(photogroup.find("Distortion/P2").text) if photogroup.find("Distortion/P2") is not None else 0.0

            # Create a unique key for this camera configuration
            camera_key = (height, width, focal_length_pixels, px, py, k1, k2, k3, p1, p2)

            camera_params[camera_key] = camera_id
            f_cam.write(f"{camera_id} FULL_OPENCV {width} {height} {focal_length_pixels} {focal_length_pixels} {px} {py} {k1} {k2} {p1} {p2} {k3} 0.0 0.0 0.0\n")

            # Process each image in this photogroup
            for photo in photogroup.findall("Photo"):
                image_path = photo.find("ImagePath").text
                image_name = os.path.basename(image_path)

                # Extract rotation matrix and convert to quaternion
                rot = photo.find("Pose/Rotation")
                R = np.array([
                    [float(rot.find("M_00").text), float(rot.find("M_01").text), float(rot.find("M_02").text)],
                    [float(rot.find("M_10").text), float(rot.find("M_11").text), float(rot.find("M_12").text)],
                    [float(rot.find("M_20").text), float(rot.find("M_21").text), float(rot.find("M_22").text)]
                ])
                qw, qx, qy, qz = rotation_matrix_to_quaternion(R)

                # Extract camera center (translation)
                center = photo.find("Pose/Center")
                tx = float(center.find("x").text)
                ty = float(center.find("y").text)
                tz = float(center.find("z").text)

                width = int(photogroup.find("ImageDimensions/Width").text)
                height = int(photogroup.find("ImageDimensions/Height").text)

                C = np.array([tx, ty, tz], dtype=np.float32)

                # Project points to this image
                #uv, visible_ids = project_points(merged_mesh, point_3d, R, C, focal_length_pixels, focal_length_pixels, px, py, k1, k2, p1, p2, k3, width, height)

                # Write to images.txt
                # if len(uv) == 0:
                #     skip += 1
                #     continue

                # find image in image_dir
                image_dir = image_name
                for dirpath, dirnames, filenames in os.walk(img_path):
                    for filename in filenames:
                        if filename == image_name:
                            image_dir = os.path.join(dirpath.split("/")[-1], filename)
                            break
                f_img.write(f"{photo_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_dir}\n")

                f_img.write("\n")

                # for idx, pid in enumerate(visible_ids.tolist()):  # .tolist() for numpy arrays
                #     track_dict[pid].append(str(photo_id) + " " + str(idx))

                photo_id += 1
                # print(f"Processed image {photo_id}: {image_name} with {len(uv)} visible points.")
            camera_id += 1
    print(f"Number of images: {photo_id}")
    print(f"Skipped {skip} images due to no visible points.")

    #save track_dict to a file
    # with open(os.path.join(output_dir, "tracks.txt"), "w") as f:
    #     f.write("# POINT3D_ID IMAGE_ID POINT2D_IDX\n")
    #     for point_id, tracks in track_dict.items():
    #         for track in tracks:
    #             f.write(f"{point_id} {track}\n")
    return track_dict

def project_points(mesh, points, R, C, fx, fy, cx, cy, k1, k2, p1, p2, k3, width, height):
    """
    Project 3D points onto the image plane and apply distortion.

    mesh: trimesh object containing the mesh
    points: 3D points to project (N, 3) numpy array
    R: Rotation matrix (3, 3) numpy array
    C: Camera center (3,) numpy array
    fx, fy: Focal lengths in x and y direction
    cx, cy: Principal point coordinates
    k1, k2, p1, p2, k3: Distortion coefficients
    width: Image width
    height: Image height
    """
    # Only keep closeby points
    points = np.array(points, dtype=np.float32)  # Ensure points are in float32 format
    # Transform points to camera coordinates
    X_cam = (R @ (points - C).T).T  # (N, 3)
    mask = X_cam[:, 2] > 0.1  # Points in front of the camera
    # Filter points that are too far away
    mask &= X_cam[:, 2] < 100 # Adjust this threshold as needed
    valid_ids = np.where(mask)[0]

    mesh = mesh.copy()
    # triangle id start
    face_mask = np.all(np.isin(mesh.faces, valid_ids), axis=1)  # Keep faces that only contain valid vertices
    mesh.faces = mesh.faces[face_mask].reshape(-1, 3)  # Reshape to ensure faces are in the correct format

    # Remove occluded points
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    for i in tqdm(range(len(X_cam))):
        ray_origins = np.array([X_cam[i]])
        ray_directions = np.array([[0, 0, -1]])
        loc, _, _ = intersector.intersects_location(ray_origins, ray_directions, multiple_hits=False)

        if loc is not None and len(loc) > 0:
            dist = np.linalg.norm(loc[0] - X_cam[i])
            d = np.linalg.norm(X_cam[i])
            if dist < 0.1:  # Adjust this threshold as needed
                mask[i] = False
                valid_ids = np.delete(valid_ids, i)  # Remove this point from valid_ids
            if d < dist:
                mask[i] = True
            else:
                mask[i] = False
                valid_ids = np.delete(valid_ids, i)  # Remove this point from valid_ids
        else:
            mask[i] = True
    X_cam = X_cam[mask]

    if len(valid_ids) == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=int)

    # Normalized coordinates
    x = X_cam[:, 0] / X_cam[:, 2]
    y = X_cam[:, 1] / X_cam[:, 2]

    # Apply distortion (FULL_OPENCV model)
    r2 = x**2 + y**2
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    x_ = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
    y_ = y * radial + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

    # Project to pixel coordinates
    u = fx * x_ + cx
    v = fy * y_ + cy

    # Check if within image bounds
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[in_bounds]
    v = v[in_bounds]
    valid_ids = valid_ids[in_bounds]

    #Only keep 10000 points
    if len(u) > 10000:
        np.random.seed(42)
        indices = np.random.choice(len(u), 10000, replace=False)
        u = u[indices]
        v = v[indices]
        valid_ids = valid_ids[indices]

    if len(u) == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=int)

    return np.column_stack((u, v)), valid_ids


def load_all_vertices_with_colors(point3D_dir):
    with  open(point3D_dir, 'r') as f:
        lines = f.readlines()

    vertices = []
    colors = []
    point_ids = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.strip().split()
        point_id = int(parts[0])
        x, y, z = map(float, parts[1:4])
        r, g, b = map(int, parts[4:7])
        vertices.append([x, y, z])
        colors.append([r, g, b])
        point_ids.append(point_id)

    return np.array(vertices), np.array(colors), np.array(point_ids)


def change_point3d(point_3d_dir, track_dict):
    with open(point_3d_dir, "r") as f:
        out = open(point_3d_dir.replace("points3D.txt", "points3D_with_image.txt"), "w")
        out.write("# 3D point list with one line of data per point:\n")
        out.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        lines = f.readlines()
        number_of_points = len(lines) - 1  # Subtract 1 for the header line
        mean_track_length = 0.0
        if track_dict:
            mean_track_length = np.mean([len(v) for v in track_dict.values()])
        out.write(f"#   Number of points: {number_of_points}, mean track length: {mean_track_length}\n")

        i = 0
        for line in tqdm(lines):
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            point_id = int(parts[0])
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            i+=1

            if point_id in track_dict:
                track_ids = " ".join(map(str, track_dict[point_id]))
                out.write(f"{i} {x} {y} {z} {r} {g} {b} 0 {track_ids}\n")
            # else:
            #     out.write(f"{i} {x} {y} {z} {r} {g} {b} 0\n")