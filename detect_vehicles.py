import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

MODEL_PATH = "yolo26s.pt"
VIDEO_PATH = "parking_lot.mp4"
OUTPUT_PATH = "parking_lot_points.mp4"

#car, motorcycle, bus, truck
VEHICLE_CLASSES = [2, 3, 5, 7]

def draw_fps(frame, fps_value):
    text = f"FPS: {fps_value: .1f}"
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
    cv2.putText(frame, text, (x, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)

def open_video_info(video_path):
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not path.is_file():
        raise IsADirectoryError(f"Path is not a file: {video_path}")

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(
                f"Failed to open video file: '{video_path}'. File may be corrupted or in an unsupported format."
            )
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Video dimensions could not be determined for file: '{video_path}'.")
    if fps <= 0:
        print(f"Warning: Video FPS could not be determined for file: '{video_path}'. Defaulting to 30 FPS.")
        fps = 30.0

    return fps, width, height

def main():
    writer = None
    exit_code = 0

    try:
        fps, width, height = open_video_info(VIDEO_PATH)

        if not Path(MODEL_PATH).exists():
            print(f"Model file not found: {MODEL_PATH}")
        try:
            model = YOLO(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from '{MODEL_PATH}': {e}") from e
        
        writer = cv2.VideoWriter(
            OUTPUT_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for output file: '{OUTPUT_PATH}'.")
        
        results = model.predict(
            source = VIDEO_PATH,
            classes=VEHICLE_CLASSES,
            conf=0.1,
            save=False,
            stream=True,
        )

        smoothed_fps = 0.0
        prev_time = time.perf_counter()

        for result in results:
            frame = result.orig_img.copy()

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv2.circle(frame, (cx, cy), radius=5, color=(0, 0, 255), thickness=-1)

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            smoothed_fps = (
                instant_fps if smoothed_fps == 0 else
                instant_fps * 0.1 + smoothed_fps * 0.9
            )
            draw_fps(frame, smoothed_fps)
            writer.write(frame)

        print(f"Detection finished. Output saved to: {OUTPUT_PATH}")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user (Ctrl+C). Exiting gracefully.")
        exit_code = 130

    except (FileNotFoundError, IsADirectoryError) as e:
        print(f"File error: {e}")
        exit_code = 1

    except RuntimeError as e:
        print(f"Error: {e}")
        exit_code = 1

    except Exception as e:
        print(f"Unexpected error ({type(e).__name__}): {e}")
        exit_code = 1

    finally:
        if writer is not None:
            writer.release()
            print("Output video closed safely.")
        cv2.destroyAllWindows()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
