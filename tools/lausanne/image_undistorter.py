# import os
#
# from tqdm import tqdm
#
# img_folder = "data/lausanne/images"
# list_subfolders = os.listdir(img_folder)
#
# for subfolder in tqdm(list_subfolders):
#     os.system(f"colmap image_undistorter \
#         --image_path data/lausanne/images/{subfolder} \
#     --input_path data/lausanne/sparse/0 \
#     --output_path data/lausanne/dense \
#     --output_type COLMAP \
#     --max_image_size 2000"
#               )

import os
import argparse
from pathlib import Path


def prepare_images(input_dir, output_dir, mode="symlink"):
    """
    Prepare images for COLMAP by collecting them from subfolders into one directory.

    Args:
        input_dir (str): Root directory containing subfolders with images.
        output_dir (str): Where to place the consolidated images.
        mode (str): "symlink" (default) or "copy".
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    # Walk through subdirectories and process images
    for root, _, files in os.walk(input_dir):
        for file in files:
            if Path(file).suffix in extensions:
                src_path = Path(root) / file
                dst_path = Path(output_dir) / file

                # Handle naming conflicts (if duplicate filenames exist)
                counter = 1
                while dst_path.exists():
                    dst_path = Path(output_dir) / f"{Path(file).stem}_{counter}{Path(file).suffix}"
                    counter += 1

                # Symlink or copy
                if mode == "symlink":
                    dst_path.symlink_to(src_path.resolve())  # Absolute symlink
                elif mode == "copy":
                    import shutil
                    shutil.copy(src_path, dst_path)
                print(f"Processed: {src_path} -> {dst_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory with subfolders of images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for consolidated images.")
    parser.add_argument("--mode", type=str, default="symlink", choices=["symlink", "copy"],
                        help="Symlink or copy files.")
    args = parser.parse_args()

    prepare_images(args.input_dir, args.output_dir, args.mode)
    print(f"Done! Images consolidated in: {args.output_dir}")
    