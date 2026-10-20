"""LLaVA single-image question answering (HF-native llava-hf checkpoints).

python llava/chat.py --image data/samples/test.jpg --prompt "Describe this image."
python llava/chat.py --model llava-hf/llava-1.5-7b-hf ...   # ~14 GB download
"""
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import device_dtype, load_image

p = argparse.ArgumentParser()
p.add_argument("--image", required=True)
p.add_argument("--prompt", default="Describe this image in detail.")
p.add_argument("--model", default="llava-hf/llava-interleave-qwen-0.5b-hf")
p.add_argument("--max-new-tokens", type=int, default=200)
args = p.parse_args()

device, dtype = device_dtype()
model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype).to(device).eval()
proc = AutoProcessor.from_pretrained(args.model)

conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}]
prompt = proc.apply_chat_template(conversation, add_generation_prompt=True)
inputs = proc(images=load_image(args.image), text=prompt, return_tensors="pt").to(device, dtype)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
print(proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
