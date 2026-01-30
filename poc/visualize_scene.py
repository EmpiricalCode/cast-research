#!/usr/bin/env python3
"""
Visualize the 3D scene by loading GLB objects, transforming them to world space,
and plotting with matplotlib.
"""
import os
import json
import numpy as np
import torch
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.transforms import quaternion_to_matrix, Transform3d, axis_angle_to_quaternion, quaternion_multiply

def compose_transform(scale, rotation, translation):
    """Same as sam3d_objects.data.dataset.tdfy.transforms_3d.compose_transform"""
    tfm = Transform3d(dtype=scale.dtype, device=scale.device)
    return tfm.scale(scale).rotate(rotation).translate(translation)

def transform_points(points, rotation, translation, scale):
    """
    Transform points from local to world space using PyTorch3D
    (Same method as SAM-3D's make_scene function)

    Args:
        points: [N, 3] numpy array of local points
        rotation: [4] quaternion [w, x, y, z] from JSON
        translation: [3] position from JSON
        scale: [3] uniform scale from JSON

    Returns:
        [N, 3] numpy array of world points
    """
    # Convert to torch tensors and flatten (handle nested JSON lists)
    points_torch = torch.tensor(points, dtype=torch.float32)
    rotation_torch = torch.tensor(rotation).flatten()
    translation_torch = torch.tensor(translation).flatten()
    scale_torch = torch.tensor(scale).flatten()

    # Apply -90° rotation around X-axis correction
    # This converts from SAM-3D's internal coordinate system to our visualization
    correction_quat = axis_angle_to_quaternion(torch.tensor([-np.pi/2, 0.0, 0.0]))
    rotation_torch = quaternion_multiply(correction_quat, rotation_torch.unsqueeze(0)).squeeze()

    # Convert quaternion to rotation matrix
    rot_matrix = quaternion_to_matrix(rotation_torch.unsqueeze(0)).squeeze()

    # Compose transformation
    transform = compose_transform(
        scale=scale_torch.unsqueeze(0),
        rotation=rot_matrix.unsqueeze(0),
        translation=translation_torch.unsqueeze(0)
    )

    # Apply transformation
    world_points_torch = transform.transform_points(points_torch)

    # Convert back to numpy
    return world_points_torch.cpu().numpy()

def load_object_points(glb_path, num_samples=5000):
    """
    Load mesh from GLB and sample points from it

    Args:
        glb_path: path to GLB file
        num_samples: number of points to sample from mesh

    Returns:
        [N, 3] array of points in local object space
    """
    mesh = trimesh.load(glb_path, force='mesh')

    # Sample points uniformly from mesh surface
    points, _ = trimesh.sample.sample_surface(mesh, num_samples)

    return points

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    sam3d_dir = os.path.join(project_root, "output/sam3d_results")
    positions_path = os.path.join(sam3d_dir, "positions.json")

    # Load positions metadata
    print(f"Loading positions from {positions_path}")
    with open(positions_path, 'r') as f:
        positions = json.load(f)

    print(f"Found {len(positions)} objects")

    # Collect all transformed points
    all_points = []
    all_colors = []

    # Color palette for different objects
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, (obj_name, transform) in enumerate(positions.items()):
        # Extract object index from name (e.g., "object_0" -> 0)
        obj_idx = int(obj_name.split('_')[1])
        glb_path = os.path.join(sam3d_dir, f"object_{obj_idx}.glb")

        if not os.path.exists(glb_path):
            print(f"Warning: {glb_path} not found, skipping")
            continue

        print(f"Loading {obj_name} from {glb_path}")

        # Load points in local space
        local_points = load_object_points(glb_path, num_samples=5000)

        # Transform to world space
        rotation = transform['rotation']
        translation = transform['translation']
        scale = transform['scale']

        world_points = transform_points(local_points, rotation, translation, scale)

        all_points.append(world_points)

        # Assign color to this object
        color = colors[idx % len(colors)]
        all_colors.append(np.tile(color[:3], (len(world_points), 1)))

        print(f"  Transformed {len(world_points)} points")
        print(f"  Local bounds: [{local_points.min(axis=0)}, {local_points.max(axis=0)}]")
        print(f"  World bounds: [{world_points.min(axis=0)}, {world_points.max(axis=0)}]")
        print(f"  Rotation: {rotation}")
        print(f"  Translation: {translation}")
        print(f"  Scale: {scale}")

    # Concatenate all points
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    print(f"\nTotal points in scene: {len(all_points)}")

    # Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(
        all_points[:, 0],
        all_points[:, 1],
        all_points[:, 2],
        c=all_colors,
        s=1,
        alpha=0.6
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scene - Transformed Objects')

    # Set view angle (elevation, azimuth)
    ax.view_init(elev=20, azim=45)

    # Equal aspect ratio
    max_range = np.array([
        all_points[:, 0].max() - all_points[:, 0].min(),
        all_points[:, 1].max() - all_points[:, 1].min(),
        all_points[:, 2].max() - all_points[:, 2].min()
    ]).max() / 2.0

    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Save
    output_path = os.path.join(project_root, "output/scene_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to {output_path}")

    plt.show()

if __name__ == "__main__":
    main()
