"""
Automatic image labeling using RAM + Grounding DINO + SAM
1. RAM generates tags for the image
2. Grounding DINO detects bounding boxes for those tags
3. SAM segments each detected box
"""
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
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.2
IOU_THRESHOLD = 0.4
# ====================================


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


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. Load image
    print("Loading image...")
    image_pil, image = load_image(IMAGE_PATH)
    W, H = image_pil.size

    # 2. RAM: Generate tags
    print("Running RAM to generate tags...")
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ram_transform = TS.Compose([TS.Resize((384, 384)), TS.ToTensor(), normalize])

    ram_model = ram(pretrained=RAM_CHECKPOINT, image_size=384, vit='swin_l')
    ram_model.eval().to(DEVICE)

    ram_image = ram_transform(image_pil.resize((384, 384))).unsqueeze(0).to(DEVICE)
    res = inference_ram(ram_image, ram_model)
    tags = res[0].replace(' |', ',')
    print(f"Tags: {tags}")

    # 3. Grounding DINO: Detect boxes for tags
    print("Running Grounding DINO...")
    grounding_model = load_grounding_dino(GROUNDING_DINO_CONFIG, GROUNDING_DINO_CHECKPOINT, DEVICE)
    boxes_filt, scores, pred_phrases = get_grounding_output(
        grounding_model, image, tags, BOX_THRESHOLD, TEXT_THRESHOLD, DEVICE
    )

    # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    # Apply NMS
    boxes_filt = boxes_filt.cpu()
    print(f"Before NMS: {boxes_filt.shape[0]} boxes")
    nms_idx = torchvision.ops.nms(boxes_filt, scores, IOU_THRESHOLD).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    print(f"After NMS: {boxes_filt.shape[0]} boxes")

    # 4. SAM: Segment each box
    print("Running SAM...")
    sam_predictor = SamPredictor(build_sam(checkpoint=SAM_CHECKPOINT).to(DEVICE))

    image_cv = cv2.imread(IMAGE_PATH)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image_cv)

    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image_cv.shape[:2]).to(DEVICE)
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    # 5. Save masks as RGBA PNGs
    print("Saving masks...")
    os.makedirs(MASKS_DIR, exist_ok=True)

    for idx, mask in enumerate(masks):
        # Get mask as boolean array (H, W)
        mask_np = mask.cpu().numpy().squeeze()  # Remove batch and channel dims if present

        # Create RGBA image
        rgba = np.zeros((mask_np.shape[0], mask_np.shape[1], 4), dtype=np.uint8)
        rgba[..., :3] = image_cv  # RGB channels from original image
        rgba[..., 3] = (mask_np * 255).astype(np.uint8)  # Alpha channel = mask

        # Save as PNG
        mask_path = os.path.join(MASKS_DIR, f"{idx}.png")
        Image.fromarray(rgba, mode='RGBA').save(mask_path)

    print(f"Saved {len(masks)} masks to {MASKS_DIR}")

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
