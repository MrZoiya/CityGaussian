import numpy as np
from tqdm import tqdm


def add_offset_to_3d_points(input_path, offset):
    output_path =  input_path.replace("points3D.txt", "points3D_with_offset.txt")

    with open(input_path, "r") as f, open(output_path, "w") as out:
        i = 0
        for line in tqdm(f):
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            x, y, z = map(float, parts[1:4])
            x += offset[0]
            y += offset[1]
            z += offset[2]
            r, g, b = map(int, parts[4:7])
            out.write(f"{i} {x} {y} {z} {r} {g} {b} 0\n")
            i += 1

def normalize_coordinates(image_dir, output_dir, point_3d_dir):

    with open(image_dir, "r") as f:
        lines = f.readlines()
    image_coords = np.array([])
    i = 0
    for l in tqdm(lines):
        if l.startswith("#"):
            continue
        parts = l.strip().split()
        if parts[0] != str(i):
            continue
        i += 1
        tx, ty, tz = map(float, parts[5:8])
        image_coords = np.append(image_coords, [tx, ty, tz])

    with open(point_3d_dir, "r") as f:
        lines = f.readlines()
    points = []
    for l in tqdm(lines):
        if l.startswith("#"):
            continue
        parts = l.strip().split()
        if len(parts) < 7:
            continue
        x, y, z = map(float, parts[1:4])
        points.append([x, y, z])