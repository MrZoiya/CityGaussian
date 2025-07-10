import numpy as np
from    vispy import app, scene
from plyfile import PlyData


def visualize_gaussian_splats(ply_path):
    # Load PLY file
    plydata = PlyData.read(ply_path)

    # Extract Gaussian parameters
    xyz = np.vstack([plydata['vertex']['x'],
                     plydata['vertex']['y'],
                     plydata['vertex']['z']]).T
    opacity = plydata['vertex']['opacity']
    scale = plydata['vertex']['scale']
    rotation = plydata['vertex']['rotation']
    sh_features = plydata['vertex']['f_dc_0']  # Spherical harmonics

    # Create Vispy canvas
    canvas = scene.SceneCanvas(keys='interactive', bgcolor='black')
    view = canvas.central_widget.add_view()

    # Create scatter plot (simplified visualization)
    scatter = scene.visuals.Markers()
    scatter.set_data(xyz, edge_color=None, face_color=(1, 1, 1, opacity), size=10)
    view.add(scatter)

    # Configure camera
    view.camera = 'turntable'
    view.camera.fov = 45

    # Add axes
    axis = scene.visuals.XYZAxis(parent=view.scene)

    canvas.show()
    app.run()


visualize_gaussian_splats('outputs/middle_colmap_coarse/input.ply')