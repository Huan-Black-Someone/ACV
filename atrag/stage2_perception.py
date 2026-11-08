import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageOps

try:
    import cv2
except ImportError:
    cv2 = None

from .stage0_filter import TelemetryRecord


def apply_clahe_rgb(rgb_arr: np.ndarray, clip_limit: float = 2.5, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    if cv2 is not None:
        lab = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_eq = clahe.apply(l_chan)
        lab_eq = cv2.merge((l_eq, a_chan, b_chan))
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    pil_im = Image.fromarray(rgb_arr)
    return np.asarray(ImageOps.equalize(pil_im), dtype=np.uint8)


def extract_bilateral_canny_roi(
    rgb_arr: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
    canny_low: int = 40,
    canny_high: int = 130,
) -> np.ndarray:
    if cv2 is not None:
        smoothed = cv2.bilateralFilter(rgb_arr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        gray = cv2.cvtColor(smoothed, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, canny_low, canny_high)
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        overlay[edges > 0] = [0, 255, 128]
        return overlay

    gray = (0.299 * rgb_arr[:, :, 0] + 0.587 * rgb_arr[:, :, 1] + 0.114 * rgb_arr[:, :, 2]).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def build_transformed_composite_image(rec: TelemetryRecord, panel_size: int = 384) -> Image.Image:
    with Image.open(rec.image_path) as im:
        pil_rgb = im.convert("RGB")
        rgb_arr = np.asarray(pil_rgb, dtype=np.uint8)

    h, w = rgb_arr.shape[:2]
    xmin, ymin, xmax, ymax = rec.bbox
    bw, bh = max(8, xmax - xmin), max(8, ymax - ymin)
    pad_x, pad_y = int(0.15 * bw), int(0.15 * bh)
    x0, y0 = max(0, xmin - pad_x), max(0, ymin - pad_y)
    x1, y1 = min(w, xmax + pad_x), min(h, ymax + pad_y)

    roi_rgb = rgb_arr[y0:y1, x0:x1]
    if roi_rgb.size == 0:
        roi_rgb = rgb_arr

    panel1 = pil_rgb.copy()
    draw = ImageDraw.Draw(panel1)
    draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 40, 40), width=max(2, int(min(w, h) * 0.008)))
    panel1 = panel1.resize((panel_size, panel_size), Image.Resampling.BICUBIC)

    clahe_roi = apply_clahe_rgb(roi_rgb)
    panel2 = Image.fromarray(clahe_roi).resize((panel_size, panel_size), Image.Resampling.BICUBIC)

    edge_roi = extract_bilateral_canny_roi(clahe_roi)
    panel3 = Image.fromarray(edge_roi).resize((panel_size, panel_size), Image.Resampling.BICUBIC)

    composite = Image.new("RGB", (panel_size * 3, panel_size))
    composite.paste(panel1, (0, 0))
    composite.paste(panel2, (panel_size, 0))
    composite.paste(panel3, (panel_size * 2, 0))
    return composite


def build_structured_vlm_prompt(rec: TelemetryRecord, is_training_kb: bool = True, dataset_name: str = "fod_a") -> str:
    xmin, ymin, xmax, ymax = rec.bbox
    roi_area_pct = ((xmax - xmin) * (ymax - ymin)) / max(1.0, float(rec.width * rec.height)) * 100.0
    flags_str = ", ".join(rec.quality_flags) if rec.quality_flags else "NONE (Clean Signal)"
    class_hint = f"Category Context: {rec.label}\n" if (is_training_kb and rec.label != "Unknown") else ""

    return (
        "You are an Evidence-Grounded Vision-Language Analyst executing structured input-augmented perception:\n"
        "  z_i = phi_CV(I_i)  and  y_i = VLM(I_i, T_i, z_i, p)\n\n"
        "Input Visual Panels (I_i and Transformed T_i):\n"
        "  - [Left Panel - I_i]     : Original image with red bounding box indicating the Region of Interest (RoI).\n"
        "  - [Middle Panel - T_i^1] : RoI enhanced via CLAHE adaptive contrast equalization.\n"
        "  - [Right Panel - T_i^2]  : Bilateral-filtered Canny structural edge map.\n\n"
        f"{class_hint}"
        "Classical CV Telemetry Vector z_i = [q_i, b_i, c_i, e_i, d_i]:\n"
        f"  - q_i (Laplacian Sharpness Variance) : {rec.q_laplacian_var:.2f}\n"
        f"  - b_i (Mean Luminance Brightness)    : {rec.b_mean_brightness:.2f}\n"
        f"  - c_i (RMS Contrast)                 : {rec.c_rms_contrast:.2f}\n"
        f"  - e_i (Canny Edge Density in RoI)    : {rec.e_canny_edge_density:.4f}\n"
        f"  - d_i (64-bit DCT-pHash)             : {rec.d_dct_phash_hex}\n"
        f"  - Spatial RoI Box & Area Ratio       : ({xmin}, {ymin}, {xmax}, {ymax}) covering {roi_area_pct:.2f}% of frame\n"
        f"  - Environmental / Context Condition  : {rec.weather}, {rec.light} | Quality Flags: {flags_str}\n\n"
        "Respond STRICTLY in JSON format with the following 4 required sections:\n"
        "```json\n"
        "{\n"
        '  "observable_description": "<concise, strictly factual description of what is directly visible in I_i and T_i>",\n'
        '  "visual_attributes": {\n'
        '    "geometric_contour": "<exact shape, boundary, and aspect ratio verified by T_i^2 Canny edge map>",\n'
        '    "surface_material_and_color": "<color, reflectivity, texture, and material cues visible in T_i^1 CLAHE panel>",\n'
        '    "structural_parts": "<observable sub-components such as threads, hex head, flat strip, or uniform surface>"\n'
        "  },\n"
        '  "uncertainty_assessment": {\n'
        '    "confidence_level": "<HIGH | MEDIUM | LOW>",\n'
        '    "limiting_factors": "<impact of small RoI area, motion blur q_i, lighting b_i, or low contrast c_i>"\n'
        "  },\n"
        '  "unsupported_conclusions": [\n'
        '    "<specific assumptions or fine-grained claims that CANNOT be verified from the current visual resolution/telemetry>"\n'
        "  ],\n"
        '  "diagnostic_inference": {\n'
        '    "predicted_class": "<inferred category supported by observable evidence>",\n'
        '    "reasoning": "<evidence-grounded justification linking visual_attributes and z_i to the category>"\n'
        "  }\n"
        "}\n"
        "```"
    )


class TriModalPerceptionEngine:
    def __init__(self, panel_size: int = 384):
        self.panel_size = panel_size

    def create_composite(self, record: TelemetryRecord) -> Image.Image:
        return build_transformed_composite_image(record, panel_size=self.panel_size)

    def create_prompt(self, record: TelemetryRecord, is_training: bool = True, dataset_name: str = "fod_a") -> str:
        return build_structured_vlm_prompt(record, is_training_kb=is_training, dataset_name=dataset_name)

    @staticmethod
    def parse_structured_response(response_text: str) -> Dict[str, Any]:
        match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = response_text.strip()
        try:
            return json.loads(json_str)
        except Exception:
            return {
                "observable_description": response_text.strip(),
                "visual_attributes": {},
                "uncertainty_assessment": {"confidence_level": "UNKNOWN"},
                "unsupported_conclusions": [],
                "diagnostic_inference": {},
            }
