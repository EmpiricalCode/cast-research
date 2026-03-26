"""
Automatic image labeling using SAM3
1. Generate tags for the image (RAM, Qwen2-VL, or GPT)
2. SAM3 performs text-grounded segmentation for each tag
"""
import argparse
import os
import sys
import numpy as np
import torch
import torchvision
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Add project root to path for cast library
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "sam3"))

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from cast.segmentation import (
    get_qwen2vl_tags,
    get_gpt_tags,
    get_ram_tags,
    show_mask,
    show_label,
)

# ============== CONFIG ==============
IMAGE_PATH = "image.jpg"
DEFAULT_OUTPUT_DIR = "output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model checkpoints
SAM3_CHECKPOINT = "./sam3-checkpoint/sam3.pt"
RAM_CHECKPOINT = "./ram_swin_large_14m.pth"

# Thresholds
IOU_THRESHOLD = 0.5
SAM3_CONFIDENCE = 0.5

# Qwen2-VL (llama.cpp)
LLAMA_CPP_BIN = "./llama.cpp/build/bin/llama-mtmd-cli"
QWEN2_VL_MODEL_PATH = "./Qwen_Qwen2.5-VL-7B-Instruct-Q5_K_M.gguf"
QWEN2_VL_MMPROJ_PATH = "./mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf"
# ====================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic image labeling using SAM3")
    parser.add_argument("--tagger", type=str, choices=["ram", "qwen", "gpt"], default="ram",
                        help="Tag generation model: 'ram' (default) or 'qwen' (Qwen2-VL-7B)")
    parser.add_argument("--image", type=str, default=IMAGE_PATH,
                        help=f"Path to input image (default: {IMAGE_PATH})")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, "output.jpg")
    MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")

    print(f"Using device: {DEVICE}")

    # 1. Load image
    print("Loading image...")
    image_pil = Image.open(args.image).convert("RGB")
    W, H = image_pil.size

    # 2. Generate tags using selected model
    if args.tagger == "qwen":
        print("Running Qwen2-VL to generate tags...")
        tags = get_qwen2vl_tags(os.path.abspath(args.image), LLAMA_CPP_BIN, QWEN2_VL_MODEL_PATH, QWEN2_VL_MMPROJ_PATH)
    elif args.tagger == "gpt":
        print("Running GPT to generate tags...")
        tags = get_gpt_tags(os.path.abspath(args.image))
    else:
        print("Running RAM to generate tags...")
        tags = get_ram_tags(image_pil, RAM_CHECKPOINT, DEVICE)

    print(f"Tags: {tags}")

    # Filter out forbidden words
    FORBIDDEN_WORDS = {'sky'}

    # Parse and deduplicate tags
    tag_list = [t.strip() for t in tags.replace('.', ',').split(',') if t.strip()]

    # Filter out tags containing forbidden words
    filtered_tags = []
    for tag in tag_list:
        words = tag.lower().split()
        if not any(word in FORBIDDEN_WORDS for word in words):
            filtered_tags.append(tag)
        else:
            print(f"  Filtering out: {tag}")

    tag_list = filtered_tags

    # Deduplicate
    seen = set()
    unique_tags = []
    for tag in tag_list:
        if tag.lower() not in seen:
            seen.add(tag.lower())
            unique_tags.append(tag)
    print(f"SAM3 input tags: {unique_tags}")

    # 3. SAM3: Text-grounded segmentation
    print("Loading SAM3...")
    sam3_model = build_sam3_image_model(device=DEVICE, checkpoint_path=SAM3_CHECKPOINT, load_from_HF=False)
    processor = Sam3Processor(sam3_model, device=DEVICE, confidence_threshold=SAM3_CONFIDENCE)

    image_cv = cv2.imread(args.image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    print("Setting image...")
    state = processor.set_image(image_pil)

    all_masks = []
    all_boxes = []
    all_scores = []
    all_phrases = []

    for tag in unique_tags:
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(tag, state=state)

        if "masks" in state and state["masks"].shape[0] > 0:
            tag_masks = state["masks"]    # (N, 1, H, W) bool
            tag_boxes = state["boxes"]    # (N, 4) [x0,y0,x1,y1] px
            tag_scores = state["scores"]  # (N,)

            nms_idx = torchvision.ops.nms(tag_boxes, tag_scores, IOU_THRESHOLD)
            tag_masks = tag_masks[nms_idx]
            tag_boxes = tag_boxes[nms_idx]
            tag_scores = tag_scores[nms_idx]

            for i in range(tag_masks.shape[0]):
                all_masks.append(tag_masks[i])
                all_boxes.append(tag_boxes[i])
                all_scores.append(tag_scores[i].item())
                all_phrases.append(f"{tag}({tag_scores[i].item():.2f})")

    if all_masks:
        masks = torch.stack(all_masks)
        boxes_filt = torch.stack(all_boxes).cpu()
        pred_phrases = all_phrases
    else:
        masks = torch.zeros((0, 1, H, W), dtype=torch.bool)
        boxes_filt = torch.zeros((0, 4))
        pred_phrases = []

    print(f"Total detections: {len(masks)}")

    # Save bounding box visualization
    print("Saving bounding box visualization...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig_bb, ax_bb = plt.subplots(1, figsize=(10, 10))
    ax_bb.imshow(Image.open(args.image))
    for i in range(boxes_filt.size(0)):
        x1, y1, x2, y2 = boxes_filt[i].tolist()
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='lime', facecolor='none')
        ax_bb.add_patch(rect)
        ax_bb.text(x1, y1 - 4, pred_phrases[i], fontsize=8, color='white',
                   bbox=dict(facecolor='lime', alpha=0.7, edgecolor='none', pad=1))
    ax_bb.axis('off')
    bb_path = os.path.join(OUTPUT_DIR, "bounding_boxes.jpg")
    fig_bb.savefig(bb_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
    plt.close(fig_bb)
    print(f"Saved bounding boxes to {bb_path}")

    # Deduplicate masks based on containment
    print("Deduplicating masks based on containment...")
    CONTAINMENT_THRESHOLD = 0.9  # If 90% of a mask is inside another, it's likely a duplicate

    masks_to_skip = set()  # Track which masks to skip

    for i in range(len(masks)):
        if i in masks_to_skip:
            continue

        mask_i = masks[i].cpu().numpy().squeeze().astype(bool)
        size_i = mask_i.sum()

        if size_i == 0:
            masks_to_skip.add(i)
            continue

        for j in range(len(masks)):
            if i == j or j in masks_to_skip:
                continue

            mask_j = masks[j].cpu().numpy().squeeze().astype(bool)
            size_j = mask_j.sum()

            if size_j == 0:
                continue

            # Calculate what % of mask_i is contained in mask_j
            intersection = (mask_i & mask_j).sum()
            containment = intersection / size_i

            # If mask_i is mostly contained in mask_j, remove the smaller one
            if containment > CONTAINMENT_THRESHOLD:
                if size_i < size_j:
                    # i is smaller and contained in j -> remove i
                    print(f"  Removing {pred_phrases[i]} ({containment:.1%} contained in {pred_phrases[j]}, smaller)")
                    masks_to_skip.add(i)
                    break
                else:
                    # i is larger but j is contained in i -> remove j
                    print(f"  Removing {pred_phrases[j]} ({containment:.1%} contained in {pred_phrases[i]}, smaller)")
                    masks_to_skip.add(j)

    # Build the kept masks/boxes/phrases lists
    masks_to_keep = []
    boxes_to_keep = []
    phrases_to_keep = []
    for i in range(len(masks)):
        if i not in masks_to_skip:
            masks_to_keep.append(masks[i])
            boxes_to_keep.append(boxes_filt[i])
            phrases_to_keep.append(pred_phrases[i])

    masks = torch.stack(masks_to_keep) if masks_to_keep else torch.zeros((0, 1, masks.shape[2], masks.shape[3]))
    boxes_filt = torch.stack(boxes_to_keep) if boxes_to_keep else torch.zeros((0, 4))
    pred_phrases = phrases_to_keep
    print(f"After deduplication: {len(masks)} masks retained")

    # 5. Save masks as RGBA PNGs
    print("Saving masks...")
    os.makedirs(MASKS_DIR, exist_ok=True)

    MIN_MASK_AREA = 0  # Minimum pixels for a valid mask
    saved_count = 0
    for idx, (mask, phrase) in enumerate(zip(masks, pred_phrases)):
        # Get mask as boolean array (H, W)
        mask_np = mask.cpu().numpy().squeeze()  # Remove batch and channel dims if present

        # Skip empty or tiny masks
        mask_area = mask_np.sum()
        if mask_area < MIN_MASK_AREA:
            print(f"  Skipping mask {idx} ({phrase}): too small ({mask_area} pixels)")
            continue

        # Create RGBA image
        rgba = np.zeros((mask_np.shape[0], mask_np.shape[1], 4), dtype=np.uint8)
        rgba[..., :3] = image_cv  # RGB channels from original image
        rgba[..., 3] = (mask_np * 255).astype(np.uint8)  # Alpha channel = mask

        # Save as PNG
        mask_path = os.path.join(MASKS_DIR, f"{saved_count}.png")
        Image.fromarray(rgba, mode='RGBA').save(mask_path)
        saved_count += 1

    print(f"Saved {saved_count} masks to {MASKS_DIR} (skipped {len(masks) - saved_count} empty/tiny masks)")

    # 6. Visualize and save
    print("Saving visualization...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(10, 10))
    plt.imshow(image_cv)

    # Store colors for each mask
    colors = []
    for mask in masks:
        color = show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        colors.append(color)

    # Draw index labels using distance transform to find best position inside mask
    for idx, mask in enumerate(masks):
        show_label(mask.cpu().numpy(), plt.gca(), str(idx), color=colors[idx])

    plt.axis('off')
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=300, pad_inches=0.0)
    print(f"Saved visualization to {OUTPUT_PATH}")

    # Print detected objects
    print("\nDetected objects:")
    for phrase in pred_phrases:
        print(f"  - {phrase}")
