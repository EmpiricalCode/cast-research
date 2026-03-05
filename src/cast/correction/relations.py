import re
import json
import base64
from pathlib import Path
from openai import OpenAI

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "graph_relation.txt"


def extract_relations_from_image(image_path, object_ids, prompt_path=None):
    """
    Use GPT to analyze an image and extract object relations as JSON.

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

    with open(prompt_path, 'r') as f:
        prompt = f.read()

    # Inject object IDs into the prompt so the model doesn't hallucinate extra objects
    object_list = ", ".join(str(i) for i in object_ids)
    prompt = f"The objects in this image are labeled: {object_list}. ONLY use these object IDs in your output.\n\n{prompt}"

    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    ext = Path(image_path).suffix.lower()
    media_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

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

    # Extract JSON from response
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())

    return json.loads(content)
