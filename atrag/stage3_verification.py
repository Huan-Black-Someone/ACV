import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor


def fallback_sent_tokenize(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def strip_technical_headers(text: str) -> str:
    header_re = re.compile(r"^\s*\[[^\]\n]*\]\s*:?\s*$")
    inline_re = re.compile(
        r"\[(?:Description|Single-VLM[^\]\n]*|Branch-1[^\]\n]*|QWEN2VL Observations|BLIP2 Observations|LLAVA Observations)\]\s*:?",
        flags=re.IGNORECASE,
    )
    kept_lines = [ln for ln in text.splitlines() if not header_re.match(ln)]
    return inline_re.sub(" ", "\n".join(kept_lines)).strip()


class CLIPSemanticVerifier:
    def __init__(
        self,
        model_id: str = "openai/clip-vit-large-patch14",
        threshold: float = 0.20,
        device: Optional[str] = None,
    ):
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        try:
            self.model = CLIPModel.from_pretrained(model_id, torch_dtype=dtype).to(self.device)
        except Exception:
            self.device = "cpu"
            self.model = CLIPModel.from_pretrained(model_id).to(self.device)

        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def verify_sentence(
        self,
        image: Image.Image,
        sentence: str,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[bool, float]:
        clean_sent = sentence.strip()
        if not clean_sent:
            return False, 0.0

        target_img = image
        if roi_bbox is not None:
            w, h = image.size
            xmin, ymin, xmax, ymax = roi_bbox
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmax), min(h, ymax)
            if xmax > xmin and ymax > ymin:
                target_img = image.crop((xmin, ymin, xmax, ymax))

        inputs = self.processor(
            text=[clean_sent],
            images=target_img,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        img_embed = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        txt_embed = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)

        sim = float(torch.sum(img_embed * txt_embed, dim=-1).item())
        passed = sim >= self.threshold
        return passed, sim

    def filter_text(
        self,
        image_path: Path,
        text: str,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        with Image.open(image_path) as im:
            rgb_img = im.convert("RGB")

        cleaned = strip_technical_headers(text)
        sentences = fallback_sent_tokenize(cleaned)
        if not sentences:
            return {"kept_text": "", "kept_count": 0, "total_count": 0, "retention_rate": 0.0}

        kept = []
        scores = []
        for s in sentences:
            passed, score = self.verify_sentence(rgb_img, s, roi_bbox=roi_bbox)
            scores.append(score)
            if passed:
                kept.append(s)

        kept_text = " ".join(kept)
        return {
            "kept_text": kept_text,
            "kept_count": len(kept),
            "total_count": len(sentences),
            "retention_rate": len(kept) / max(1, len(sentences)),
            "scores": scores,
        }
