from pathlib import Path
import torch
import yaml
import os

# 1. Ortamın Google Colab olup olmadığını otomatik kontrol et
IN_COLAB = Path("/content").exists()

# project path otomatically arranged by the script, but can be overridden by the user
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.getenv("PARKING_DATA_DIR", PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

VIDEO_PATH = PROJECT_ROOT / config["paths"]["video_path"]

# YAML dosyasını oku
with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

MODEL_PATH = PROJECT_ROOT / config["model"]["path"]

CONFIDENCE_THRESHOLD = config["inference"]["conf"]
MODEL_IOU = config["inference"]["iou"]
IMGSZ = config["inference"]["imgsz"]

SLICE_HEIGHT = config["sahi"]["slice_height"]
SLICE_WIDTH = config["sahi"]["slice_width"]
OVERLAP_HEIGHT_RATIO = config["sahi"]["overlap_height_ratio"]
OVERLAP_WIDTH_RATIO = config["sahi"]["overlap_width_ratio"]

POSTPROCESS_TYPE = config["sahi"]["postprocess_type"]
POSTPROCESS_MATCH_METRIC = config["sahi"]["postprocess_match_metric"]
POSTPROCESS_MATCH_THRESHOLD = config["sahi"]["postprocess_match_threshold"]

WARMUP_FRAMES = config["benchmark"]["warmup_frames"]
BENCHMARK_FRAMES = config["benchmark"]["measured_frames"]

# COCO sınıf ID'leri: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = {2, 3, 5, 7}

# cuda ortama göre seçilir
configured_device = config["model"].get("device", "auto")

if configured_device == "auto":
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = configured_device

# Deney ID'si ve dosya yolları
EXPERIMENT_ID = (
    f"{Path(MODEL_PATH).stem}"
    f"_conf{CONFIDENCE_THRESHOLD}"
    f"_iou{MODEL_IOU}"
    f"_imgsz{IMGSZ}"
)

VIDEO_OUTPUT_DIR = OUTPUT_DIR / "runs" 
BENCHMARK_OUTPUT_DIR = OUTPUT_DIR / "results" / "benchmarks"

VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = VIDEO_OUTPUT_DIR / f"parking_lot_boxes_{EXPERIMENT_ID}.mp4"
SUMMARY_PATH = BENCHMARK_OUTPUT_DIR / f"benchmark_summary_{EXPERIMENT_ID}.csv"
FRAME_RESULTS_PATH = BENCHMARK_OUTPUT_DIR / f"benchmark_frames_{EXPERIMENT_ID}.csv"