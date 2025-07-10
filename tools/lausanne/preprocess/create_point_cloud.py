import glob

import trimesh
from tqdm import tqdm


def color_point_cloud(OBJ_PATH, output_path):
    obj_files = glob.glob(f"{OBJ_PATH}/*.obj")
    # Initialize output
    all_colored_points = []
    point_id = 1
    meshes = []
    with open(output_path, 'w') as out:
        for obj_file in tqdm(obj_files):
            # Load the scene
            scene = trimesh.load(obj_file, process=False)
            if isinstance(scene, trimesh.Scene):
                # Get all meshes from the scene
                meshes_scene = list(scene.geometry.values())
                if not meshes:
                    print(f"Skipping {obj_file}: no meshes found")
                    continue
                meshes.extend(meshes_scene)
            else:
                meshes.append(scene)

        n_mesh = 0
        for mesh in tqdm(meshes):
            # Get vertex colors
            try:
                vertex_colors = mesh.visual.to_color().vertex_colors
            except:
                #print(f"Skipping {obj_file}: could not extract vertex colors")
                continue

            # Sample points from the mesh
            points = mesh.vertices

            for i in range(len(points)):
                x, y, z = points[i]
                r, g, b, a = vertex_colors[i]
                out.write(f"{point_id} {x} {y} {z} {r} {g} {b} 0\n")
                point_id += 1

            n_mesh += 1
    print(f"Colored point {point_id}")