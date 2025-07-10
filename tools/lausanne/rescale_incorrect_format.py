import os
import json
import numpy as np
import torch
import xml.etree.ElementTree as ET
import sqlite3

from PIL import Image

import internal.utils.colmap as colmap

basic_path = "/mnt/Data/yanns/CityGaussian/data/lausanne"
file_images = "/mnt/Data/yanns/CityGaussian/data/lausanne_9000/images"


tree = ET.parse(os.path.join(basic_path, "cameras.xml"))
root = tree.getroot()

block = root.find("Block")
cameras = []
image_paths = []
image_names = []
poses = []
centers = []
image_camera_ids = []
for photogroup in block.findall(".//Photogroup"):
    imageDimensions = photogroup.find("ImageDimensions")
    width = int(imageDimensions.find("Width").text)
    height = int(imageDimensions.find("Height").text)
    focal_length = float(photogroup.find("FocalLength").text)
    sensor_size = float(photogroup.find("SensorSize").text)

    focal_length_in_pixel = focal_length / sensor_size * width

    cameras.append({
        "width": width,
        "height": height,
        "focal_length": focal_length_in_pixel,
        "principal_point": (
            float(photogroup.find("PrincipalPoint/x").text), float(photogroup.find("PrincipalPoint/y").text)),
        "distortion": {i.tag: float(i.text) for i in photogroup.find("Distortion")} if photogroup.find("Distortion") is not None else {
            "K1": 0.0, "K2": 0.0, "P1": 0.0, "P2": 0.0
        },
    })
    camera_idx = len(cameras) - 1

    for photo in photogroup.findall("Photo"):
        rotation = list(photo.find("Pose/Rotation"))
        center = list(photo.find("Pose/Center"))
        if rotation[-1].text == "false":
            continue
        if center[-1].text == "false":
            continue

        path_split = photo.find("ImagePath").text.split("/")[-2:]
        path = "/".join(path_split)
        image_paths.append(path)
        image_names.append(path_split[-1])
        poses.append([float(i.text) for i in rotation])
        centers.append([float(i.text) for i in center])
        image_camera_ids.append(camera_idx)

for path, id in zip(image_paths, image_camera_ids):
    image_path = os.path.join(file_images, path)
    if not os.path.exists(image_path):
        continue
    image = Image.open(image_path)
    camera_id = id

    if image.size[0] != cameras[camera_id]["width"] or image.size[1] != cameras[camera_id]["height"]:
        print(f"Image {path} has incorrect size: {image.size}, expected: {cameras[camera_id]['width']}x{cameras[camera_id]['height']}")
        image = image.resize((cameras[camera_id]["width"], cameras[camera_id]["height"]))
        image.save(image_path)



