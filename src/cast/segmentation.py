import subprocess
import sys
import numpy as np
import torch
import cv2
from PIL import Image

import torchvision.transforms as TS
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from ram.models import ram
from ram import inference_ram


def get_qwen2vl_tags(image_path, llama_cpp_bin, model_path, mmproj_path):
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
- Do NOT include background elements like "wall", "ceiling", or "window" unless they are prominent objects in the image.
- YOU MUST INCLUDE EVERY OBJECT WITHIN THE IMAGE!!! TRY TO INCLUDE COLOR AND MATERIAL OF OBJECTS IF POSSIBLE.
- DO NOT INCLUDE THE SKY!!! DO NOT INCLUDE THE BACKGROUND!!!
- DO NOT INCLUDE OBJECTS THAT ARE ONLY PARTIALLY VISIBLE AND ARE CUT OFF SIGNIFICANTLY!!!
- ALWAYS OPT FOR GENERIC DESCIPTORS WITH MATERIALS (e.g., "wooden table" instead of "dining table", "metal chair" instead of "office chair")

Example output: sand, couch, lamp, coffee table, book, plant, window, rug"""

    result = subprocess.run(
        [
            llama_cpp_bin,
            "-m", model_path,
            "--mmproj", mmproj_path,
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


def load_image_for_gdino(image_path):
    """Load and preprocess image for Grounding DINO."""
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
        # Ensure each RGB value is at least 100/255 to avoid dark colors
        rgb = np.random.uniform(100/255, 1.0, 3)
        color = np.concatenate([rgb, np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return color[:3]  # Return RGB color without alpha


def show_label(mask, ax, label, color=None):
    # Use distance transform to find the point furthest from mask edges
    mask_np = mask.squeeze().astype(np.uint8)
    dist = cv2.distanceTransform(mask_np, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist)
    cx, cy = max_loc[0], max_loc[1]  # x, y in image coordinates

    if color is None:
        color = [0, 1, 0]
    ax.text(cx, cy, label, fontsize=9, color=color, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='square', facecolor='black', alpha=0.8, pad=0.1))


def get_ram_tags(image_pil, ram_checkpoint, device):
    """
    Use RAM to generate tags for the image.
    Returns a comma-separated string of object tags.
    """
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ram_transform = TS.Compose([TS.Resize((384, 384)), TS.ToTensor(), normalize])

    ram_model = ram(pretrained=ram_checkpoint, image_size=384, vit='swin_l')
    ram_model.eval().to(device)

    ram_image = ram_transform(image_pil.resize((384, 384))).unsqueeze(0).to(device)
    res = inference_ram(ram_image, ram_model)
    tags = res[0].replace(' |', ',')
    return tags
