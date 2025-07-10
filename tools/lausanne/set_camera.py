from collections import defaultdict

import numpy as np


def map_camera_id(cursor, image_dir):
    """
    Maps camera IDs to image paths in the database.
    Args:
        image_dir (str): Directory containing images.
    Returns:
        dict: Mapping of image paths to camera IDs.
    """
    with open(image_dir) as f:
        lines = f.readlines()

    path_id_map = {}

    for i, line in enumerate(lines):
        if line.startswith("#"):
            continue
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        image_path = parts[-1]
        camera_id = int(parts[-2])
        # Finds the camera ID for the image path in the database
        database_camera_id = cursor.execute("SELECT camera_id FROM images WHERE name = ?", (image_path,)).fetchone()

        if not database_camera_id:
            print(f"Image {image_path} not found in database.")
            continue

        path_id_map[camera_id] = database_camera_id[0]

    return path_id_map


def change_camera(
        cursor,
        camera_id,
        model=None,
        width=None,
        height=None,
        params=None,
):
    if model is not None:
        cursor.execute(
            "UPDATE cameras SET model = ? WHERE camera_id = ?",
            (model, camera_id)
        )
    if width is not None:
        cursor.execute(
            "UPDATE cameras SET width = ? WHERE camera_id = ?",
            (width, camera_id)
        )
    if height is not None:
        cursor.execute(
            "UPDATE cameras SET height = ? WHERE camera_id = ?",
            (height, camera_id)
        )
    if params is not None:
        params_str = " ".join(map(str, params))
        cursor.execute(
            "UPDATE cameras SET params = ? WHERE camera_id = ?",
            (params_str, camera_id)
        )
    cursor.connection.commit()


import sqlite3

conn = sqlite3.connect("data/lausanne/database_middle_BU.db")
cursor = conn.cursor()

image_dir = "data/lausanne/sparse/images.txt"  # Path to the images file
path_id_map = map_camera_id(cursor, image_dir)

camera_dir = "data/lausanne/sparse/cameras.txt"
with open(camera_dir, "r") as f:
    lines = f.readlines()

updated = []
for line in lines:
    if line.startswith("#"):
        updated.append(line)
        continue
    parts = line.strip().split()
    if len(parts) < 5:
        updated.append(line)
        continue
    camera_id = int(parts[0])
    model = parts[1]
    width = int(parts[2])
    height = int(parts[3])
    params = list(map(float, parts[4:]))

    if camera_id not in path_id_map:
        print(f"Camera ID {camera_id} not found in path_id_map.")
        updated.append(line)
        continue
    #change_camera(cursor, path_id_map[camera_id], model=model, width=width, height=height, params=params)
    updated.append(f"{path_id_map[camera_id]} {model} {width} {height} {' '.join(map(str, params))}\n")

with open(camera_dir, "w") as f:
    f.writelines(updated)

with open("data/lausanne/sparse/images.txt", "r") as f:
    lines = f.readlines()

updated_images = []
for line in lines:
    if line.startswith("#"):
        updated_images.append(line)
        continue
    parts = line.strip().split()
    if len(parts) < 7:
        updated_images.append(line)
        continue
    image_path = parts[-1]
    camera_id = int(parts[-2])

    updated_images.append(f"{' '.join(parts[:-2])} {path_id_map[camera_id]} {image_path}\n")
with open("data/lausanne/sparse/images.txt", "w") as f:
    f.writelines(updated_images)