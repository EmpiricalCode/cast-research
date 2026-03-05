import numpy as np
import torch
from pytorch3d.transforms import (
    quaternion_to_matrix,
    Transform3d,
    axis_angle_to_quaternion,
    quaternion_multiply,
    rotation_6d_to_matrix,
)


def apply_x90_correction(quaternion):
    """
    Apply the -90 degree X-axis correction that converts from SAM-3D's
    internal coordinate system to the visualization coordinate system.

    Args:
        quaternion: [4] torch tensor [w, x, y, z]

    Returns:
        [4] corrected quaternion
    """
    correction_quat = axis_angle_to_quaternion(torch.tensor([-np.pi/2, 0.0, 0.0]))
    return quaternion_multiply(correction_quat, quaternion.unsqueeze(0)).squeeze()


def apply_inverse_x90_correction(quaternion):
    """
    Inverse of apply_x90_correction. Used after optimization to convert back
    to SAM-3D's coordinate system for saving.

    Args:
        quaternion: [4] torch tensor [w, x, y, z] in visualization space

    Returns:
        [4] quaternion in SAM-3D's coordinate system
    """
    inverse_correction_quat = axis_angle_to_quaternion(torch.tensor([np.pi/2, 0.0, 0.0]))
    return quaternion_multiply(inverse_correction_quat, quaternion.unsqueeze(0)).squeeze()


def compose_transform(scale, rotation, translation):
    """Same as sam3d_objects.data.dataset.tdfy.transforms_3d.compose_transform"""
    tfm = Transform3d(dtype=scale.dtype, device=scale.device)
    return tfm.scale(scale).rotate(rotation).translate(translation)


def transform_points_quat(points, rotation, translation, scale):
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
    rotation_torch = apply_x90_correction(rotation_torch)

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


def local_to_world_6d(local_points, rotation_6d, translation, scale):
    """
    Transform points from object's local space to world space.

    Args:
        local_points: [N, 3] torch tensor in object's local space
        rotation_6d: [6] 6D rotation representation
        translation: [3] translation
        scale: [3] scale

    Returns:
        [N, 3] points in world space
    """
    # Convert 6D rotation to rotation matrix
    rot_matrix = rotation_6d_to_matrix(rotation_6d.unsqueeze(0)).squeeze()

    # Forward transform: scale -> rotate -> translate
    tfm_forward = Transform3d(device=local_points.device)
    tfm_forward = tfm_forward.scale(scale.unsqueeze(0)).rotate(rot_matrix.unsqueeze(0)).translate(translation.unsqueeze(0))

    # Apply transformation
    world_points = tfm_forward.transform_points(local_points)

    return world_points


def world_to_local_6d(world_points, target_rotation_6d, target_translation, target_scale, normalization_scale):
    """
    Transform world points to target object's local normalized space [-1, 1]³.

    Args:
        world_points: [N, 3] torch tensor
        target_rotation_6d: [6] 6D rotation representation
        target_translation: [3] translation
        target_scale: [3] scale (uniform) - world transform scale
        normalization_scale: float - scale factor from mesh normalization

    Returns:
        [N, 3] points in target's normalized SDF space
    """
    # Convert 6D rotation to rotation matrix
    rot_matrix = rotation_6d_to_matrix(target_rotation_6d.unsqueeze(0)).squeeze()

    # Forward transform: scale -> rotate -> translate
    tfm_forward = Transform3d(device=world_points.device)
    tfm_forward = tfm_forward.scale(target_scale.unsqueeze(0)).rotate(rot_matrix.unsqueeze(0)).translate(target_translation.unsqueeze(0))

    # Get inverse transform (world -> target's local GLB space)
    tfm_inverse = tfm_forward.inverse()
    points_target_local = tfm_inverse.transform_points(world_points)

    # Apply normalization to get into SDF grid space
    # The SDF was computed on a mesh normalized by dividing by normalization_scale
    points_normalized = points_target_local / normalization_scale

    return points_normalized
