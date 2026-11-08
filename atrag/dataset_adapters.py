import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List


@dataclass
class CandidateSample:
    dataset_name: str
    sample_id: str
    image_path: Path
    class_label: str
    xml_bbox_path: Optional[Path]
    weather_or_event: str
    light_or_modality: str
    companion_text: str
    existing_caption: str


def load_foda_samples(base_dir: Path, limit: int = 0) -> List[CandidateSample]:
    voc_dir = base_dir / "datasets" / "datasets" / "FODPascalVOCFormat-V.2.1" / "VOC2007"
    train_csv = voc_dir / "split_datasets" / "train.csv"
    train_img_dir = voc_dir / "split_datasets" / "train" / "images"
    xml_dir = voc_dir / "Annotations"
    env_csv = voc_dir / "ImageSets" / "Main" / "CategorizationData" / "FOD_categorization_annotations.csv"

    env_lookup = {}
    if env_csv.exists():
        with open(env_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                stem = Path(row.get("File", "").strip()).stem
                if stem:
                    env_lookup[stem] = (
                        row.get("Weather", "dry").strip().lower(),
                        row.get("Light", "bright").strip().lower(),
                    )

    samples = []
    if train_csv.exists():
        with open(train_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                stem = Path(row.get("image_name", "").strip()).stem
                img_path = train_img_dir / f"{stem}.jpg"
                if not img_path.exists():
                    continue

                w, l = env_lookup.get(stem, ("dry", "bright"))
                xml_path = xml_dir / f"{stem}.xml"
                samples.append(CandidateSample(
                    dataset_name="fod_a",
                    sample_id=stem,
                    image_path=img_path,
                    class_label=row.get("label", "Unknown").strip(),
                    xml_bbox_path=xml_path if xml_path.exists() else None,
                    weather_or_event=w,
                    light_or_modality=l,
                    companion_text="",
                    existing_caption="",
                ))
                if 0 < limit <= len(samples):
                    break
    return samples


def get_dataset_adapter(dataset_name: str, base_dir: Path, limit: int = 0) -> List[CandidateSample]:
    name = dataset_name.lower().replace("-", "_")
    if "fod" in name:
        return load_foda_samples(base_dir, limit=limit)
    raise ValueError(f"Unsupported dataset: {dataset_name}")
