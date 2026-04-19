import re
import json
import base64
from pathlib import Path
from openai import OpenAI

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "graph_relation.txt"
SCENE_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "scene_description.txt"


def _build_scaffolding(object_ids):
    """Build the scaffolding with per-object descriptions followed by JSON."""
    lines = []
    for a in object_ids:
        lines.append(f"Object {a}: ...")
    lines.append("")
    scaffold = {"scene_description": {}}
    for a in object_ids:
        scaffold["scene_description"][str(a)] = {}
        for b in object_ids:
            if str(a) != str(b):
                scaffold["scene_description"][str(a)][str(b)] = "..."
    lines.append(json.dumps(scaffold, indent=2))
    return "\n".join(lines)


def _encode_image(image_path):
    """Encode image as base64 and determine media type."""
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    ext = Path(image_path).suffix.lower()
    media_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    return image_data, media_type


def _extract_json(content):
    """Extract JSON from GPT response text."""
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return json.loads(content)


def _pass1_scene_description(client, image_data, media_type, object_ids):
    """Pass 1: Get broad contact descriptions for every object pair."""
    with open(SCENE_PROMPT_PATH, 'r') as f:
        template = f.read()

    scaffolding = _build_scaffolding(object_ids)
    object_list = ", ".join(str(i) for i in object_ids)
    prompt = f"The objects in this image are labeled: {object_list}.\n\n{template.replace('{scaffolding}', scaffolding)}"

    response = client.chat.completions.create(
        model="gpt-5.2",
        temperature=0,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_data}"}}
            ]
        }]
    )

    content = response.choices[0].message.content.strip()
    return _extract_json(content)


def _pass2_hard_relations(client, image_data, media_type, object_ids, scene_description, prompt_path):
    """Pass 2: Generate structured relations — same as original, with scene description as extra context."""
    with open(prompt_path, 'r') as f:
        relation_prompt = f.read()

    object_list = ", ".join(str(i) for i in object_ids)
    scene_desc_text = json.dumps(scene_description, indent=2)

    # Same as original prompt injection, but with scene description prepended as context
    prompt = (
        f"The objects in this image are labeled: {object_list}. ONLY use these object IDs in your output.\n\n"
        f"Here is a preliminary scene description of spatial relationships between all object pairs. "
        f"Use this as context to ensure you do not miss any relationships:\n\n"
        f"{scene_desc_text}\n\n"
        f"{relation_prompt}"
    )

    response = client.chat.completions.create(
        model="gpt-5.2",
        temperature=0,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_data}"}}
            ]
        }]
    )

    content = response.choices[0].message.content.strip()
    return _extract_json(content)


def extract_relations_from_image(image_path, object_ids, prompt_path=None):
    """
    Use GPT to analyze an image and extract object relations as JSON.
    Two-pass approach: first extracts broad scene descriptions for all pairs,
    then uses those as context to generate structured relations.

    Args:
        image_path: path to the labeled scene image
        object_ids: list of object IDs present in the image
        prompt_path: path to the prompt text file (defaults to bundled prompt)

    Returns:
        dict with "relations" key containing the object relation graph
    """
    if prompt_path is None:
        prompt_path = PROMPT_PATH

    client = OpenAI()
    image_data, media_type = _encode_image(image_path)

    # Pass 1: broad scene description
    print("  Pass 1: Extracting scene descriptions...")
    scene_result = _pass1_scene_description(client, image_data, media_type, object_ids)
    scene_description = scene_result.get("scene_description", scene_result)
    print(f"  Pass 1 result: {json.dumps(scene_description, indent=2)}")

    # Pass 2: structured relations using scene description as context
    print("  Pass 2: Extracting structured relations...")
    return _pass2_hard_relations(client, image_data, media_type, object_ids, scene_description, prompt_path)
