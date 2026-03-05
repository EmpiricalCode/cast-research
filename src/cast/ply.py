import numpy as np


def write_ply(points, colors, output_path):
    """
    Write a colored point cloud to an ASCII PLY file.

    Args:
        points: [N, 3] numpy array of xyz coordinates
        colors: [N, 3] numpy array of RGB values in [0, 1] range
        output_path: path to write the PLY file
    """
    colors_uint8 = (colors * 255).astype(np.uint8)

    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for pt, col in zip(points, colors_uint8):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {col[0]} {col[1]} {col[2]}\n")
