import csv
import cv2
import torch
from core.settings import (
    DEVICE,
    FRAME_RESULTS_PATH,
    SUMMARY_PATH,
    MODEL_PATH,
    VIDEO_PATH,
    WARMUP_FRAMES,
)

def sync_device():
    """GPU işlemlerinin bitmesini bekler (Doğru zamanlama için)."""
    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

def draw_fps(frame, fps_value):
    """Kare üzerine anlık FPS değerini yazar."""
    text = f"FPS: {fps_value:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness, margin = 0.8, 2, 10

    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

    x = frame.shape[1] - text_w - margin
    y = margin + text_h

    cv2.rectangle(
        frame,
        (x - 5, y - text_h - 5),
        (x + text_w + 5, y + baseline + 5),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )

def open_video_info(video_path):
    """Videonun özelliklerini (fps, genişlik, yükseklik, toplam kare) okur."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if width <= 0 or height <= 0:
        raise RuntimeError("Could not determine video dimensions.")

    if fps <= 0:
        print("Warning: Invalid source FPS. Using 30 FPS.")
        fps = 30.0

    return fps, width, height, total_frames

def save_benchmark_results(latencies, frame_numbers, measured_count):
    """Benchmark ölçümlerini ve özetini CSV dosyalarına kaydeder."""
    if not latencies:
        print("No benchmark measurements available.")
        return

    total_time = sum(latencies)
    average_latency_ms = total_time / measured_count * 1000
    average_fps = measured_count / total_time

    # Kare bazlı gecikme sürelerini kaydet
    with open(FRAME_RESULTS_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["frame_number", "latency_ms", "fps_from_latency"],
        )
        writer.writeheader()
        for frame_number, latency in zip(frame_numbers, latencies):
            writer.writerow({
                "frame_number": frame_number,
                "latency_ms": round(latency * 1000, 3),
                "fps_from_latency": round(1 / latency, 3),
            })

    # Özet raporu kaydet
    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model", "device", "video", "warmup_frames",
                "measured_frames", "average_latency_ms", "average_fps", "measurement_scope"
            ],
        )
        writer.writeheader()
        writer.writerow({
            "model": MODEL_PATH,
            "device": DEVICE,
            "video": VIDEO_PATH.name,
            "warmup_frames": WARMUP_FRAMES,
            "measured_frames": measured_count,
            "average_latency_ms": round(average_latency_ms, 3),
            "average_fps": round(average_fps, 3),
            "measurement_scope": "BGR-to-RGB conversion + SAHI prediction; excludes drawing and video writing",
        })

    print("\n========== BENCHMARK RESULTS ==========")
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Measured frames: {measured_count}")
    print(f"Average latency: {average_latency_ms:.2f} ms/frame")
    print(f"Average FPS: {average_fps:.2f}")
    print(f"Summary saved: {SUMMARY_PATH}")
    print(f"Per-frame results saved: {FRAME_RESULTS_PATH}")