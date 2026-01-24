"""
Automatic image labeling using RAM + Grounding DINO + SAM
1. RAM generates tags for the image
2. Grounding DINO detects bounding boxes for those tags
3. SAM segments each detected box
"""
import argparse
import os
import numpy as np
import torch
import torchvision
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import torchvision.transforms as TS

import sys
sys.path.insert(0, '..')  # Add parent directory to path

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything
from segment_anything import build_sam, SamPredictor

# RAM
from ram.models import ram
from ram import inference_ram

# Qwen2-VL (via llama.cpp CLI)
import subprocess

# ============== CONFIG ==============
IMAGE_PATH = "image.jpg"
OUTPUT_DIR = "output"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "output.jpg")
MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model checkpoints - UPDATE THESE PATHS
GROUNDING_DINO_CONFIG = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "./groundingdino_swint_ogc.pth"
SAM_CHECKPOINT = "./sam_vit_h_4b8939.pth"
RAM_CHECKPOINT = "./ram_swin_large_14m.pth"

# Thresholds
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.2
IOU_THRESHOLD = 0.3

# Qwen2-VL (llama.cpp)
LLAMA_CPP_BIN = "./llama.cpp/build/bin/llama-mtmd-cli"
QWEN2_VL_MODEL_PATH = "./Qwen_Qwen2.5-VL-7B-Instruct-Q5_K_M.gguf"
QWEN2_VL_MMPROJ_PATH = "./mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf"
# ====================================


def get_qwen2vl_tags(image_path):
    """
    Use Qwen2-VL-7B via llama.cpp CLI to identify all distinct objects in an image.
    Returns a comma-separated string of object tags.
    """
    prompt = """List all distinct object types visible in this image.
Rules:v
- Output ONLY a comma-separated list of object names, nothing else
- List each object type only ONCE, even if there are multiple instances (e.g., 3 chairs = just "chair")
- NEVER repeat any object name - each word should appear only once in your output
- Do NOT include sub-parts of objects (e.g., if there's a lamp, don't also list "lampshade" separately)
- Be specific but not overly detailed (e.g., "chair" not "wooden dining chair with cushion")
- Stop after listing each unique object once
- IMPORTANT: Include the ground/floor surface type (e.g., sand, grass, dirt, concrete, wooden floor, tile).
- Do NOT include background elements like "wall", "ceiling", or "window" unless they are prominent objects in the image.
- YOU MUST INCLUDE EVERY OBJECT WITHIN THE IMAGE!!! TRY TO INCLUDE COLOR AND MATERIAL OF OBJECTS IF POSSIBLE.
- DO NOT INCLUDE THE SKY!!!
- ALWAYS OPT FOR GENERIC DESCIPTORS WITH MATERIALS (e.g., "wooden table" instead of "dining table", "metal chair" instead of "office chair")

Example output: sand, couch, lamp, coffee table, book, plant, window, rug"""

    import sys

    result = subprocess.run(
        [
            LLAMA_CPP_BIN,
            "-m", QWEN2_VL_MODEL_PATH,
            "--mmproj", QWEN2_VL_MMPROJ_PATH,
            "--image", image_path,
            "-p", prompt,
            "-n", "128",  # Reduced for faster inference
            "-ngl", "99",  # Offload all layers to GPU
        ],
        stdout=subprocess.PIPE,
        stderr=sys.stderr,  # Show progress on terminal
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"llama.cpp failed with return code {result.returncode}")

    return result.stdout.strip()


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_grounding_dino(config_path, checkpoint_path, device):
    args = SLConfig.fromfile(config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, device):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."

    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    # Filter by threshold
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    # Get phrases
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        # Clean up repeated words (e.g., "beach towel beach towel" -> "beach towel")
        words = pred_phrase.split()
        if len(words) > 1:
            # Find the shortest repeating pattern
            for pattern_len in range(1, len(words) // 2 + 1):
                pattern = words[:pattern_len]
                is_repeat = True
                for i in range(pattern_len, len(words)):
                    if words[i] != pattern[i % pattern_len]:
                        is_repeat = False
                        break
                if is_repeat:
                    pred_phrase = ' '.join(pattern)
                    break
        pred_phrases.append(pred_phrase + f"({logit.max().item():.2f})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


def show_mask(mask, ax, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0 - 5, label, fontsize=8, color='white',
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))


def get_ram_tags(image_pil, device):
    """
    Use RAM to generate tags for the image.
    Returns a comma-separated string of object tags.
    """
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ram_transform = TS.Compose([TS.Resize((384, 384)), TS.ToTensor(), normalize])

    ram_model = ram(pretrained=RAM_CHECKPOINT, image_size=384, vit='swin_l')
    ram_model.eval().to(device)

    ram_image = ram_transform(image_pil.resize((384, 384))).unsqueeze(0).to(device)
    res = inference_ram(ram_image, ram_model)
    tags = res[0].replace(' |', ',')
    return tags


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic image labeling using Grounding DINO + SAM")
    parser.add_argument("--tagger", type=str, choices=["ram", "qwen"], default="ram",
                        help="Tag generation model: 'ram' (default) or 'qwen' (Qwen2-VL-7B)")
    parser.add_argument("--image", type=str, default=IMAGE_PATH,
                        help=f"Path to input image (default: {IMAGE_PATH})")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    # 1. Load image
    print("Loading image...")
    image_pil, image = load_image(args.image)
    W, H = image_pil.size

    # 2. Generate tags using selected model
    if args.tagger == "qwen":
        print("Running Qwen2-VL to generate tags...")
        tags = get_qwen2vl_tags(os.path.abspath(args.image))
    else:
        print("Running RAM to generate tags...")
        tags = get_ram_tags(image_pil, DEVICE)
        
    print(f"Tags: {tags}")

    # Filter out forbidden words
    FORBIDDEN_WORDS = {'sky'}

    # Convert comma-separated tags to period-separated format for Grounding DINO
    # Grounding DINO expects tags like "cat . dog . chair" not "cat, dog, chair"
    # Also deduplicate tags while preserving order
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
    grounding_tags = ' . '.join(unique_tags)
    print(f"Grounding DINO input: {grounding_tags}")

    # 3. Grounding DINO: Detect boxes for each tag separately to avoid cross-tag confusion
    print("Running Grounding DINO...")
    grounding_model = load_grounding_dino(GROUNDING_DINO_CONFIG, GROUNDING_DINO_CHECKPOINT, DEVICE)

    all_boxes = []
    all_scores = []
    all_phrases = []

    for tag in unique_tags:
        boxes, scores_t, phrases = get_grounding_output(
            grounding_model, image, tag, BOX_THRESHOLD, TEXT_THRESHOLD, DEVICE
        )
        for i in range(len(boxes)):
            all_boxes.append(boxes[i])
            all_scores.append(scores_t[i].item())
            # Use the original tag name instead of parsed phrase
            all_phrases.append(f"{tag}({scores_t[i].item():.2f})")

    if all_boxes:
        boxes_filt = torch.stack(all_boxes)
        scores = torch.tensor(all_scores)
        pred_phrases = all_phrases
    else:
        boxes_filt = torch.zeros((0, 4))
        scores = torch.tensor([])
        pred_phrases = []

    # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    # Apply class-aware NMS (batched_nms) - only suppresses boxes of the same class
    boxes_filt = boxes_filt.cpu()
    print(f"Before NMS: {boxes_filt.shape[0]} boxes")

    # Create class IDs from tag names (boxes with same tag get same class ID)
    tag_to_id = {tag: i for i, tag in enumerate(unique_tags)}
    class_ids = torch.tensor([tag_to_id[phrase.rsplit('(', 1)[0]] for phrase in pred_phrases])

    nms_idx = torchvision.ops.batched_nms(boxes_filt, scores, class_ids, IOU_THRESHOLD).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    print(f"After NMS: {boxes_filt.shape[0]} boxes")

    # 4. SAM: Segment each box
    print("Running SAM...")
    sam_predictor = SamPredictor(build_sam(checkpoint=SAM_CHECKPOINT).to(DEVICE))

    image_cv = cv2.imread(args.image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image_cv)

    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image_cv.shape[:2]).to(DEVICE)
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

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

    # Build the kept masks list
    masks_to_keep = []
    phrases_to_keep = []
    for i in range(len(masks)):
        if i not in masks_to_skip:
            masks_to_keep.append(masks[i])
            phrases_to_keep.append(pred_phrases[i])

    masks = torch.stack(masks_to_keep) if masks_to_keep else torch.zeros((0, 1, masks.shape[2], masks.shape[3]))
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

    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)

    plt.axis('off')
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=300, pad_inches=0.0)
    print(f"Saved visualization to {OUTPUT_PATH}")

    # Print detected objects
    print("\nDetected objects:")
    for phrase in pred_phrases:
        print(f"  - {phrase}")
