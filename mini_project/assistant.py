"""Multimodal Image Assistant: image + question -> answer (LLaVA), plus CLIP retrieval -> LLaVA explanation.

    python mini_project/assistant.py --image lab/images/ex11_vqa/kitchen1.jpg --question "What is the man doing?"
    python mini_project/assistant.py --query "a person playing football" --image-dir lab/images/ex07_retrieval --k 3
"""
import argparse
import os

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, LlavaNextForConditionalGeneration, LlavaNextProcessor

LLAVA_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
CLIP_ID = "openai/clip-vit-base-patch32"


class MultimodalAssistant:
    """Pass already-loaded models (e.g. from the notebook) or let the class load them."""

    def __init__(self, llava=None, llava_proc=None, clip=None, clip_proc=None, device="cuda", dtype=torch.bfloat16):
        self.device, self.dtype = device, dtype
        self.llava = llava or LlavaNextForConditionalGeneration.from_pretrained(LLAVA_ID, dtype=dtype).to(device).eval()
        self.llava_proc = llava_proc or LlavaNextProcessor.from_pretrained(LLAVA_ID)
        self.clip, self.clip_proc = clip, clip_proc

    # ---- basic architecture: IMAGE + QUESTION -> LLaVA -> ANSWER ----
    @torch.no_grad()
    def answer(self, image, question, max_new_tokens=200):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = self.llava_proc.apply_chat_template(conv, add_generation_prompt=True)
        inputs = self.llava_proc(images=image, text=prompt, return_tensors="pt").to(self.device, self.dtype)
        out = self.llava.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.llava_proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # ---- bonus: USER QUERY -> CLIP -> TOP-K IMAGES -> LLaVA -> EXPLANATION ----
    def _load_clip(self):
        if self.clip is None:
            self.clip = CLIPModel.from_pretrained(CLIP_ID, dtype=self.dtype).to(self.device).eval()
            self.clip_proc = CLIPProcessor.from_pretrained(CLIP_ID)

    @torch.no_grad()
    def retrieve(self, query, image_dir, k=3):
        self._load_clip()
        files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        images = [Image.open(os.path.join(image_dir, f)).convert("RGB") for f in files]
        img = self.clip.get_image_features(**self.clip_proc(images=images, return_tensors="pt").to(self.device)).pooler_output
        txt = self.clip.get_text_features(**self.clip_proc(text=[query], return_tensors="pt").to(self.device)).pooler_output
        img = img / img.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        scores = (img @ txt.T).squeeze(1)
        top = scores.argsort(descending=True)[:k].tolist()
        return [(files[i], images[i], scores[i].item()) for i in top]

    def retrieve_and_explain(self, query, image_dir, k=3):
        results = []
        for fname, image, score in self.retrieve(query, image_dir, k):
            question = f'A search engine returned this image for the query "{query}". ' \
                       f"Does the image match the query? Answer yes or no, then explain in one sentence."
            results.append({"image": fname, "clip_cosine": round(score, 3), "llava_explanation": self.answer(image, question, 80)})
        return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--question", default="Describe this image.")
    ap.add_argument("--query")
    ap.add_argument("--image-dir")
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    assistant = MultimodalAssistant()
    if a.query:
        for r in assistant.retrieve_and_explain(a.query, a.image_dir, a.k):
            print(f"{r['image']}  cos={r['clip_cosine']}\n  {r['llava_explanation']}\n")
    else:
        print(assistant.answer(a.image, a.question))
