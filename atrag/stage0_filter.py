from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Tuple
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class TelemetryRecord:
    image_id: str
    image_path: str
    label: str
    width: int
    height: int
    bbox: Tuple[int, int, int, int]
    weather: str
    light: str
    q_laplacian_var: float
    b_mean_brightness: float
    c_rms_contrast: float
    e_canny_edge_density: float
    d_dct_phash_hex: str
    passed_gate: bool
    reject_reason: str
    quality_flags: List[str]
    composite_quality_score: float

    def to_dict(self):
        return asdict(self)


def compute_dct_phash_64bit(gray_img: np.ndarray) -> str:
    if cv2 is not None:
        resized = cv2.resize(gray_img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        dct_mat = cv2.dct(resized)
    else:
        pil_32 = Image.fromarray(gray_img).resize((32, 32), Image.Resampling.BILINEAR)
        arr = np.asarray(pil_32, dtype=np.float32)
        n = 32
        k = np.arange(n)[:, None]
        x = np.arange(n)[None, :]
        basis = np.cos(np.pi * (2 * x + 1) * k / (2.0 * n))
        dct_mat = basis @ arr @ basis.T

    low_freq = dct_mat[:8, :8].flatten()
    ac_coeffs = low_freq[1:]
    med = np.median(ac_coeffs)
    bits = (low_freq > med).astype(np.uint8)
    bit_str = "".join(str(b) for b in bits)
    return f"{int(bit_str, 2):016x}"


def phash_hamming_similarity(hash_a: str, hash_b: str) -> float:
    try:
        xor_val = int(hash_a, 16) ^ int(hash_b, 16)
        hamming_dist = bin(xor_val).count("1")
        return 1.0 - (hamming_dist / 64.0)
    except Exception:
        return 0.0


def compute_laplacian_variance(gray_img: np.ndarray) -> float:
    if cv2 is not None:
        lap = cv2.Laplacian(gray_img, cv2.CV_64F)
        return float(lap.var())
    g = gray_img.astype(np.float64)
    lap = (
        g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    )
    return float(lap.var())


def compute_canny_edge_density(gray_img: np.ndarray, low_thresh: int = 50, high_thresh: int = 150) -> float:
    if gray_img.size == 0:
        return 0.0
    if cv2 is not None:
        edges = cv2.Canny(gray_img, low_thresh, high_thresh)
        return float(np.count_nonzero(edges)) / float(edges.size)
    g = gray_img.astype(np.float32)
    gx = np.abs(g[:, 2:] - g[:, :-2])
    gy = np.abs(g[2:, :] - g[:-2, :])
    mag = np.hypot(gx[1:-1, :], gy[:, 1:-1])
    return float(np.mean(mag > low_thresh))


def infer_salient_roi_bbox(gray_img: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
    if gray_img.size > 0:
        if cv2 is not None:
            edges = cv2.Canny(gray_img, 50, 150)
            ys, xs = np.nonzero(edges)
        else:
            g = gray_img.astype(np.float32)
            gx = np.abs(g[:, 2:] - g[:, :-2])
            gy = np.abs(g[2:, :] - g[:-2, :])
            mag = np.hypot(gx[1:-1, :], gy[:, 1:-1])
            ys, xs = np.nonzero(mag > 50.0)

        if len(xs) >= 32 and len(ys) >= 32:
            xmin = max(0, int(np.percentile(xs, 10)))
            xmax = min(width, int(np.percentile(xs, 90)))
            ymin = max(0, int(np.percentile(ys, 10)))
            ymax = min(height, int(np.percentile(ys, 90)))
            if (xmax - xmin) >= 16 and (ymax - ymin) >= 16:
                return (xmin, ymin, xmax, ymax)

    pad_x, pad_y = int(width * 0.2), int(height * 0.2)
    return (pad_x, pad_y, max(pad_x + 1, width - pad_x), max(pad_y + 1, height - pad_y))


def parse_voc_xml_bbox(
    xml_path: Optional[Path], width: int, height: int, gray_img: Optional[np.ndarray] = None
) -> Tuple[int, int, int, int]:
    if xml_path is not None and xml_path.exists():
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            obj = root.find("object")
            if obj is not None:
                bb = obj.find("bndbox")
                if bb is not None:
                    xmin = max(0, int(float(bb.findtext("xmin", "0"))))
                    ymin = max(0, int(float(bb.findtext("ymin", "0"))))
                    xmax = min(width, int(float(bb.findtext("xmax", str(width)))))
                    ymax = min(height, int(float(bb.findtext("ymax", str(height)))))
                    if xmax > xmin and ymax > ymin:
                        return (xmin, ymin, xmax, ymax)
        except Exception:
            pass
    if gray_img is not None:
        return infer_salient_roi_bbox(gray_img, width, height)
    pad_x, pad_y = int(width * 0.2), int(height * 0.2)
    return (pad_x, pad_y, max(pad_x + 1, width - pad_x), max(pad_y + 1, height - pad_y))


class DefensiveFilter:
    def __init__(
        self,
        min_std_degeneracy: float = 2.0,
        min_laplacian_sharpness: float = 15.0,
        min_brightness: float = 15.0,
        max_brightness: float = 245.0,
        min_rms_contrast: float = 8.0,
    ):
        self.min_std_degeneracy = min_std_degeneracy
        self.min_laplacian_sharpness = min_laplacian_sharpness
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_rms_contrast = min_rms_contrast

    def process_sample(
        self,
        image_path: Path,
        label: str,
        xml_path: Optional[Path] = None,
        weather: str = "Dry",
        light: str = "Bright",
    ) -> TelemetryRecord:
        img_id = image_path.stem

        if not image_path.exists() or image_path.stat().st_size == 0:
            return TelemetryRecord(
                image_id=img_id, image_path=str(image_path), label=label,
                width=0, height=0, bbox=(0, 0, 0, 0), weather=weather, light=light,
                q_laplacian_var=0.0, b_mean_brightness=0.0, c_rms_contrast=0.0,
                e_canny_edge_density=0.0, d_dct_phash_hex="0" * 16,
                passed_gate=False, reject_reason="Tier1_CorruptedOrMissingFile",
                quality_flags=["FILE_INVALID"], composite_quality_score=0.0,
            )

        try:
            with Image.open(image_path) as pil_im:
                pil_rgb = pil_im.convert("RGB")
                width, height = pil_rgb.size
                rgb_arr = np.asarray(pil_rgb, dtype=np.uint8)
        except Exception as e:
            return TelemetryRecord(
                image_id=img_id, image_path=str(image_path), label=label,
                width=0, height=0, bbox=(0, 0, 0, 0), weather=weather, light=light,
                q_laplacian_var=0.0, b_mean_brightness=0.0, c_rms_contrast=0.0,
                e_canny_edge_density=0.0, d_dct_phash_hex="0" * 16,
                passed_gate=False, reject_reason=f"Tier1_DecodeError:{type(e).__name__}",
                quality_flags=["DECODE_ERROR"], composite_quality_score=0.0,
            )

        gray = (
            0.299 * rgb_arr[:, :, 0] + 0.587 * rgb_arr[:, :, 1] + 0.114 * rgb_arr[:, :, 2]
        ).astype(np.uint8)

        b_mean = float(np.mean(gray))
        c_rms = float(np.std(gray))
        if c_rms < self.min_std_degeneracy or width < 16 or height < 16:
            return TelemetryRecord(
                image_id=img_id, image_path=str(image_path), label=label,
                width=width, height=height, bbox=(0, 0, width, height), weather=weather, light=light,
                q_laplacian_var=0.0, b_mean_brightness=round(b_mean, 4), c_rms_contrast=round(c_rms, 4),
                e_canny_edge_density=0.0, d_dct_phash_hex="0" * 16,
                passed_gate=False, reject_reason="Tier2_InformationDegeneracy",
                quality_flags=["DEGENERATE_SIGNAL"], composite_quality_score=0.0,
            )

        bbox = parse_voc_xml_bbox(xml_path, width, height, gray_img=gray)
        xmin, ymin, xmax, ymax = bbox
        roi_gray = gray[ymin:ymax, xmin:xmax]
        if roi_gray.size < 64:
            roi_gray = gray

        q_lap = compute_laplacian_variance(roi_gray)
        e_edge = compute_canny_edge_density(roi_gray)
        d_phash = compute_dct_phash_64bit(gray)

        flags = []
        if q_lap < self.min_laplacian_sharpness:
            flags.append("LOW_SHARPNESS_BLUR")
        if b_mean < self.min_brightness:
            flags.append("UNDER_EXPOSED_DARK")
        elif b_mean > self.max_brightness:
            flags.append("OVER_EXPOSED_GLARE")
        if c_rms < self.min_rms_contrast:
            flags.append("LOW_RMS_CONTRAST")

        q_norm = min(1.0, np.log1p(q_lap) / np.log1p(500.0))
        b_norm = max(0.0, 1.0 - abs(b_mean - 127.5) / 127.5)
        c_norm = min(1.0, c_rms / 64.0)
        e_norm = min(1.0, e_edge / 0.25)
        composite_q = float(0.40 * q_norm + 0.20 * b_norm + 0.20 * c_norm + 0.20 * e_norm)

        return TelemetryRecord(
            image_id=img_id,
            image_path=str(image_path),
            label=label,
            width=width,
            height=height,
            bbox=bbox,
            weather=weather,
            light=light,
            q_laplacian_var=round(q_lap, 4),
            b_mean_brightness=round(b_mean, 4),
            c_rms_contrast=round(c_rms, 4),
            e_canny_edge_density=round(e_edge, 4),
            d_dct_phash_hex=d_phash,
            passed_gate=True,
            reject_reason="",
            quality_flags=flags,
            composite_quality_score=round(composite_q, 4),
        )
