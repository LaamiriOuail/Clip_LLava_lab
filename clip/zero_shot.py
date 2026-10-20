"""CLIP zero-shot classification.

python clip/zero_shot.py --image data/samples/test.jpg --labels "a cat" "a dog" "a car"
"""
import argparse
import sys
from pathlib import Path

import torch
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import device_dtype, load_image

p = argparse.ArgumentParser()
p.add_argument("--image", required=True)
p.add_argument("--labels", nargs="+", required=True)
p.add_argument("--model", default="openai/clip-vit-base-patch32")
args = p.parse_args()

device, dtype = device_dtype()
model = CLIPModel.from_pretrained(args.model, dtype=dtype).to(device).eval()
proc = CLIPProcessor.from_pretrained(args.model)

inputs = proc(text=args.labels, images=load_image(args.image), return_tensors="pt", padding=True).to(device)
with torch.inference_mode():
    probs = model(**inputs).logits_per_image.float().softmax(-1)[0]

for label, prob in sorted(zip(args.labels, probs.tolist()), key=lambda x: -x[1]):
    print(f"{prob:6.2%}  {label}")
