"""
MoGe monocular geometry estimation with per-object point cloud extraction.
Uses MoGe to produce a full-scene point map, then applies autolabel masks
to extract individual object point clouds.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import trimesh
from scipy.spatial.distance import cdist

# Add MoGe to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MoGe"))

from moge.model.v2 import MoGeModel
import utils3d

# Add project src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, "src"))

from cast.ply import write_ply
from cast.correction.transforms import apply_x90_correction, apply_inverse_x90_correction, local_to_world_6d
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion, matrix_to_rotation_6d
from pytorch3d.ops import iterative_closest_point


def max_pairwise_distance(pts):
    """Compute the max distance between any two points in a point cloud."""
    # Use bounding box diagonal as a fast approximation-free upper bound check,
    # but compute exact max by checking distances from extreme points
    # Subsample if too large for full pairwise (N^2 memory)
    if len(pts) > 5000:
        subset = pts[np.random.default_rng(0).choice(len(pts), 5000, replace=False)]
    else:
        subset = pts
    dists = cdist(subset, subset)
    return dists.max()


def main():
    parser = argparse.ArgumentParser(description="Run MoGe and extract per-object point clouds using masks")
    parser.add_argument("--image", type=str, required=True, help="Path to original input image")
    parser.add_argument("--dir", type=str, required=True, help="Run directory (e.g. output/run-35fa7ba1)")
    parser.add_argument("--model", type=str, default="Ruicheng/moge-2-vitl-normal", help="Pretrained model name or path")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 for faster inference")
    parser.add_argument("--resolution-level", type=int, default=9, help="Resolution level 0-9 (default: 9)")
    parser.add_argument("--threshold", type=float, default=0.04, help="Edge removal threshold (default: 0.04)")
    parser.add_argument("--max-points", type=int, default=100000, help="Max points per object (default: 100000)")
    args = parser.parse_args()

    image_path = Path(args.image)
    run_dir = Path(args.dir)
    masks_dir = run_dir / "masks"
    sam3d_dir = run_dir / "sam3d_results"
    output_dir = run_dir / "moge"

    if not image_path.exists():
        print(f"Error: image not found: {image_path}")
        sys.exit(1)
    if not masks_dir.exists():
        print(f"Error: masks directory not found: {masks_dir}")
        sys.exit(1)

    positions_file = sam3d_dir / "positions.json"
    if not sam3d_dir.exists():
        print(f"Error: sam3d_results directory not found: {sam3d_dir}")
        sys.exit(1)
    if not positions_file.exists():
        print(f"Error: positions.json not found: {positions_file}")
        sys.exit(1)

    # Load transformations (same as optimize_sdf.py)
    with open(positions_file, 'r') as f:
        transformations_raw = json.load(f)

    transformations = {}
    for name, tfm in transformations_raw.items():
        target_rotation = torch.tensor(tfm['rotation'], dtype=torch.float32).flatten()
        target_translation = torch.tensor(tfm['translation'], dtype=torch.float32).flatten()
        target_scale = torch.tensor(tfm['scale'], dtype=torch.float32).flatten()

        # Apply -90° rotation around X-axis correction
        # Converts from SAM-3D's internal coordinate system to visualization coordinate system
        target_rotation = apply_x90_correction(target_rotation)

        transformations[name] = {
            "rotation": target_rotation,
            "translation": target_translation,
            "scale": target_scale,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Load model
    print(f"Loading MoGe model: {args.model}")
    model = MoGeModel.from_pretrained(args.model).to(device).eval()
    if args.fp16:
        model.half()

    # Load image
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

    # Run MoGe inference
    print(f"Running MoGe inference on {image_path} ({width}x{height})")
    output = model.infer(image_tensor, resolution_level=args.resolution_level, use_fp16=args.fp16)
    points = output["points"].cpu().numpy()   # (H, W, 3) camera-space point map
    depth = output["depth"].cpu().numpy()      # (H, W)
    valid_mask = output["mask"].cpu().numpy()  # (H, W) valid geometry mask
    intrinsics = output["intrinsics"].cpu().numpy()  # (3, 3)

    # Clean up depth edges
    edge_mask = utils3d.np.depth_map_edge(depth, rtol=args.threshold)
    valid_mask = valid_mask & ~edge_mask

    # OpenGL flip then 180° rotation around Y axis
    # OpenGL flip: (x, y, z) -> (x, -y, -z)
    # 180° around Y: (x, -y, -z) -> (-x, -y, z)
    points = points * [-1, -1, 1]

    # Normalize colors
    colors = image.astype(np.float32) / 255

    # Save full scene point cloud
    scene_valid = valid_mask.astype(bool)
    scene_points = points[scene_valid]
    scene_colors = colors[scene_valid]
    if args.max_points and len(scene_points) > args.max_points:
        indices = np.random.default_rng(42).choice(len(scene_points), args.max_points, replace=False)
        scene_points = scene_points[indices]
        scene_colors = scene_colors[indices]

    scene_path = output_dir / "scene.ply"
    print(f"Saving full scene ({len(scene_points)} points) to {scene_path}")
    write_ply(scene_points, scene_colors, str(scene_path))

    # Load masks and extract per-object point clouds
    mask_files = sorted(masks_dir.glob("*.png"), key=lambda p: int(p.stem))
    print(f"Found {len(mask_files)} masks")

    tab20 = cm.get_cmap("tab20")

    all_segmented_points = []
    all_segmented_colors = []
    all_scale_factors = []
    # Store per-object data for ICP pass
    icp_pairs = {}  # idx -> {"moge_points": np.array, "sam3d_world_points": np.array}

    rng = np.random.default_rng(42)
    for i, mask_file in enumerate(mask_files):
        idx = mask_file.stem
        # Mask is RGBA where alpha = object mask
        rgba = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
        object_mask = (rgba[:, :, 3] > 0) & valid_mask

        object_points = points[object_mask]
        object_colors = colors[object_mask]

        if len(object_points) == 0:
            print(f"  Object {idx}: no valid points, skipping")
            continue

        # Downsample if needed
        if args.max_points and len(object_points) > args.max_points:
            indices = rng.choice(len(object_points), args.max_points, replace=False)
            object_points = object_points[indices]
            object_colors = object_colors[indices]

        out_path = output_dir / f"{idx}.ply"
        print(f"  Object {idx}: {len(object_points)} points -> {out_path}")
        write_ply(object_points, object_colors, str(out_path))

        # Compare with SAM3D .glb mesh (transformed to global SAM3D space)
        glb_path = sam3d_dir / f"{idx}.glb"
        if glb_path.exists() and idx in transformations:
            mesh = trimesh.load(str(glb_path), force="mesh")
            sam3d_local_samples, _ = trimesh.sample.sample_surface(mesh, 10000)
            sam3d_local_tensor = torch.tensor(sam3d_local_samples, dtype=torch.float32)

            # Transform local → world using same method as optimize_sdf.py:
            # quaternion → rotation matrix → 6D rotation, then local_to_world_6d
            tfm = transformations[idx]
            rotation_6d = matrix_to_rotation_6d(
                quaternion_to_matrix(tfm["rotation"].unsqueeze(0)).squeeze()
            )
            sam3d_world_samples = local_to_world_6d(
                sam3d_local_tensor,
                rotation_6d,
                tfm["translation"],
                tfm["scale"],
            ).detach().numpy()

            moge_diameter = max_pairwise_distance(object_points)
            sam3d_diameter = max_pairwise_distance(sam3d_world_samples)

            if moge_diameter > 0:
                scale_factor = sam3d_diameter / moge_diameter
                all_scale_factors.append(scale_factor)
                print(f"  Object {idx} scale: {scale_factor:.6f} (SAM3D diameter: {sam3d_diameter:.4f}, MoGe diameter: {moge_diameter:.4f})")
            else:
                print(f"  Object {idx}: MoGe diameter is 0, cannot compute scale")

            icp_pairs[idx] = {
                "moge_points": object_points,
                "sam3d_world_points": sam3d_world_samples,
            }
        else:
            print(f"  Object {idx}: no matching .glb or transformations entry found")

        # Collect for color-coded segmentation PLY
        object_color = np.array(tab20(i % 20)[:3])  # RGB in [0, 1]
        seg_colors = np.broadcast_to(object_color, object_points.shape).copy()
        all_segmented_points.append(object_points)
        all_segmented_colors.append(seg_colors)

    # Save color-coded segmentation PLY
    if all_segmented_points:
        seg_points = np.concatenate(all_segmented_points)
        seg_colors = np.concatenate(all_segmented_colors)
        seg_path = output_dir / "segmented.ply"
        print(f"Saving color-coded segmentation ({len(seg_points)} points) to {seg_path}")
        write_ply(seg_points, seg_colors, str(seg_path))

    # Compute robust scale factor via IQR outlier removal
    if all_scale_factors:
        factors = np.array(all_scale_factors)
        print(f"\nScale factors ({len(factors)}): {factors}")
        print(f"  Std dev: {factors.std():.6f}")

        q1 = np.percentile(factors, 25)
        q3 = np.percentile(factors, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        inliers = factors[(factors >= lower) & (factors <= upper)]

        if len(inliers) < len(factors):
            outliers = factors[(factors < lower) | (factors > upper)]
            print(f"  IQR bounds: [{lower:.6f}, {upper:.6f}]")
            print(f"  Removed {len(factors) - len(inliers)} outliers: {outliers}")

        if len(inliers) > 0:
            avg_scale = inliers.mean()
            print(f"\nScale factor: {avg_scale:.6f}")
        else:
            print("\nWarning: all scale factors were outliers, using unfiltered mean")
            avg_scale = factors.mean()
            print(f"\nScale factor: {avg_scale:.6f}")
    else:
        print("\nWarning: no scale factors computed")
        avg_scale = None

    # ICP refinement: align SAM3D world points to scaled MoGe points per object
    if avg_scale is not None and icp_pairs:
        print(f"\n--- ICP refinement (scale={avg_scale:.6f}) ---")
        updated_transformations = {}

        for idx, pair in icp_pairs.items():
            moge_pts = pair["moge_points"]
            sam3d_pts = pair["sam3d_world_points"]

            # Scale MoGe points into SAM3D world space
            moge_scaled = moge_pts * avg_scale

            # Subsample both to at most 10000 points for ICP performance
            max_icp_points = 10000
            if len(moge_scaled) > max_icp_points:
                moge_scaled = moge_scaled[rng.choice(len(moge_scaled), max_icp_points, replace=False)]
            if len(sam3d_pts) > max_icp_points:
                sam3d_pts = sam3d_pts[rng.choice(len(sam3d_pts), max_icp_points, replace=False)]

            # ICP: source X = MoGe (partial), target Y = SAM3D (complete)
            # MoGe is partial so every source point has a valid nearest neighbor in SAM3D
            # Finds: X_moge @ R_icp + T_icp ≈ Y_sam3d
            X = torch.tensor(moge_scaled, dtype=torch.float32, device=device).unsqueeze(0)
            Y = torch.tensor(sam3d_pts, dtype=torch.float32, device=device).unsqueeze(0)

            icp_result = iterative_closest_point(
                X, Y,
                estimate_scale=False,
                max_iterations=100,
            )

            if icp_result.converged:
                print(f"  Object {idx}: ICP converged, RMSE={icp_result.rmse.item():.6f}")
            else:
                print(f"  Object {idx}: ICP did not converge, RMSE={icp_result.rmse.item():.6f}")

            # ICP gives: X_moge @ R_icp + T_icp ≈ Y_sam3d
            # We need the inverse to correct SAM3D: Y_sam3d @ R_inv + T_inv ≈ X_moge
            # R_inv = R_icp^T, T_inv = -T_icp @ R_icp^T
            R_icp = icp_result.RTs.R[0]  # (3, 3)
            T_icp = icp_result.RTs.T[0]  # (3,)

            R2 = R_icp.T
            T2 = -T_icp @ R_icp.T

            print(f"    ICP T_inv={T2.cpu().numpy()}")

            # Compose with original SAM3D transformation
            # Original: quaternion -> rotation matrix (with x90 correction already applied)
            tfm = transformations[idx]
            R1 = quaternion_to_matrix(tfm["rotation"].unsqueeze(0).to(device)).squeeze()  # (3, 3)
            T1 = tfm["translation"].to(device)  # (3,)
            s1 = tfm["scale"].to(device)  # (3,) uniform

            # Compose: s_final = s1, R_final = R1 @ R2, T_final = T1 @ R2 + T2
            s_final = s1
            R_final = R1 @ R2
            T_final = T1 @ R2 + T2

            # Convert R_final back to quaternion, undo x90 correction for saving
            q_final = matrix_to_quaternion(R_final.unsqueeze(0)).squeeze()
            q_final_uncorrected = apply_inverse_x90_correction(q_final)

            updated_transformations[idx] = {
                "rotation": q_final_uncorrected.cpu().detach().tolist(),
                "translation": T_final.cpu().detach().tolist(),
                "scale": s_final.cpu().detach().tolist(),
            }

        # Save updated positions.json
        # Merge: keep original entries, overwrite with ICP-refined ones
        with open(positions_file, 'r') as f:
            original_positions = json.load(f)

        for idx, tfm_updated in updated_transformations.items():
            original_positions[idx] = tfm_updated

        icp_positions_path = output_dir / "positions_icp.json"
        with open(icp_positions_path, 'w') as f:
            json.dump(original_positions, f, indent=2)
        print(f"\nSaved ICP-refined positions to {icp_positions_path}")

    print(f"Done. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
