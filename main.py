import time
import cv2
import torch

from core.settings import (
    OUTPUT_PATH,
    DEVICE,
    WARMUP_FRAMES,
    BENCHMARK_FRAMES,
    VEHICLE_CLASSES,
    VIDEO_PATH,
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    MODEL_IOU,
    IMGSZ,
)
from services.detector import load_model, run_inference
from utils.video_utils import (
    sync_device,
    draw_fps,
    open_video_info,
    save_benchmark_results,
)

def main():
    cap = None
    video_writer = None
    latencies = []
    measured_frame_numbers = []
    frame_count = 0
    smoothed_fps = 0.0

    try:
        print(f"Device: {DEVICE}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU unavailable. Using CPU.")

        fps, width, height, total_frames = open_video_info(VIDEO_PATH)
        print(f"Video: {width}x{height} at {fps:.2f} FPS")

        # Modeli yükle
        detection_model = load_model()

        cap = cv2.VideoCapture(str(VIDEO_PATH))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open: {VIDEO_PATH}")

        video_writer = cv2.VideoWriter(
            str(OUTPUT_PATH),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not video_writer.isOpened():
            raise RuntimeError(f"Could not create output: {OUTPUT_PATH}")

        print("Starting video processing...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            should_measure = (
                frame_count >= WARMUP_FRAMES
                and len(latencies) < BENCHMARK_FRAMES
            )

            if should_measure:
                sync_device()

            frame_start = time.perf_counter()

            # OpenCV BGR -> RGB dönüşümü
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Tahmin servisini çağır
            result = run_inference(frame_rgb, detection_model)

            # if frame_count == 1:
            #     predictor = detection_model.model.predictor

            #     print("Configured vehicle classes:", list(VEHICLE_CLASSES))
            #     print("Actual predictor classes:", predictor.args.classes)

            sync_device()
            elapsed = time.perf_counter() - frame_start

            if should_measure:
                latencies.append(elapsed)
                measured_frame_numbers.append(frame_count + 1)

            instant_fps = 1.0 / max(elapsed, 1e-9)
            smoothed_fps = (
                instant_fps
                if smoothed_fps == 0
                else 0.1 * instant_fps + 0.9 * smoothed_fps
            )

            vehicle_count = 0
            
            for prediction in result.object_prediction_list: 
                class_id = int(prediction.category.id)
                if class_id not in VEHICLE_CLASSES: # sınıf kontrolü yedek
                    continue

                x1, y1, x2, y2 = map(int, prediction.bbox.to_xyxy())
                score = float(prediction.score.value)
                class_name = prediction.category.name

                # Bounding Box Çizimi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{class_name}: {score:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale, thickness = 0.5, 1

                (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
                label_top = max(0, y1 - text_h - baseline - 4)

                cv2.rectangle(
                    frame,
                    (x1, label_top),
                    (x1 + text_w + 4, max(y1, text_h + baseline + 4) if label_top == 0 else y1),
                    (0, 255, 0),
                    -1,
                )

                text_y = max(text_h + 2, y1 - baseline - 2)
                cv2.putText(
                    frame,
                    label,
                    (x1 + 2, text_y),
                    font,
                    scale,
                    (0, 0, 0),
                    thickness,
                    cv2.LINE_AA,
                )
                vehicle_count += 1

            draw_fps(frame, smoothed_fps)
            video_writer.write(frame)
            frame_count += 1

            print(
                f"\rFrames: {frame_count}/{total_frames} | "
                f"Vehicles: {vehicle_count} | "
                f"Benchmark: {len(latencies)}/{BENCHMARK_FRAMES}",
                end="",
                flush=True,
            )

        print("\nVideo processing finished.")

    except KeyboardInterrupt:
        print("\nProcessing interrupted.")
    except Exception as error:
        print(f"\nError ({type(error).__name__}): {error}")
    finally:
        if cap is not None:
            cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()

        if latencies:
            save_benchmark_results(latencies, measured_frame_numbers, len(latencies))
        else:
            print("No benchmark frames were measured.")

        print(f"Output video: {OUTPUT_PATH}")
        print("Video resources released.")

if __name__ == "__main__":
    main()