import subprocess
import sys
import os
import base64
import numpy as np
import torch
import cv2
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

import torchvision.transforms as TS
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


def get_gpt_tags(image_path):
    """
    Use GPT via OpenAI API to identify all distinct objects in an image.
    Returns a comma-separated string of object tags.
    """
    client = OpenAI()

    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("utf-8")

    prompt = """You are part of a 3D scene reconstruction pipeline. Your job is to identify every distinct object in this image so that each one can be turned into its own 3D mesh. The full scene will be represented as a collection of these meshes, so your list must be complete and each entry should correspond to something that makes sense as a standalone mesh.

Before listing, reason about the scene: if two things are physically part of the same object (e.g., a countertop and the cabinet it is glued onto), they should be ONE entry (countertop with cabinet). If two things are distinct and separable (e.g., a book sitting on a table), they should be SEPARATE entries.

List all distinct object types visible in this image.
Rules:
- Output ONLY a comma-separated list of object names, nothing else
- List each object type only ONCE, even if there are multiple instances (e.g., 3 chairs = just "chair")
- Do NOT include sub-parts of objects (e.g., if there's a lamp, don't also list "lampshade" separately)
- Do NOT be overly detailed (e.g., "chair" not "wooden dining chair with cushion"). Keep entries as simple as possible. Combined entries should be joined using keyword "with".
- Stop after listing each unique object once
- Do NOT include background elements like "wall", "ceiling", or "window" unless they are objects fully contained in the image (IE, a window on a table).
- When describing containers with amorphous contents, group them together (e.g., "bowl with sauce" instead of "bowl, pudding". Keyword with.)
- Think about the environment as a whole before listing. For each object, ask yourself "Does it make sense for this object exist independently? Or is it physically connected to or part of another object? IE, a counter-top shouldn't be a seperate object from the counter itself. However, distinct books stacked on top of each other should be seperate.
- YOU MUST INCLUDE EVERY OBJECT WITHIN THE IMAGE!!!
- DO NOT INCLUDE THE SKY!!! DO NOT INCLUDE THE BACKGROUND!!!
- DO NOT INCLUDE OBJECTS THAT ARE ONLY PARTIALLY VISIBLE AND ARE CUT OFF SIGNIFICANTLY!!!
- BE CAREFUL NOT TO MISS A SINGLE VALID OBJECT

Example output: sand, couch, lamp, coffee table, book, plant, window, rug
"""

    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
            ],
        }],
        max_completion_tokens=256,
        temperature=0,
    )

    return response.choices[0].message.content.strip()


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
