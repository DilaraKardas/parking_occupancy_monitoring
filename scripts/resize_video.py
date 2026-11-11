import argparse
from pathlib import Path
from core.settings import PROJECT_ROOT, VIDEO_PATH
import cv2

parser = argparse.ArgumentParser(description="Resize a video to a specified width while maintaining the aspect ratio.")
parser.add_argument(
    "--input", 
    type=str, 
    default=str(VIDEO_PATH), 
    help="Input video path"
)
parser.add_argument(
    "--output", 
    type=str, 
    default=str(PROJECT_ROOT / "videos" / "parking_lot.mp4"), 
    help="Output video path"
)
parser.add_argument(
    "--width", 
    type=int, 
    default=960, 
    help="video width in pixels (default: 960)"
)

args = parser.parse_args()

input_path = args.input
output_path = args.output
target_width = args.width

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Video cannot be opened: {input_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

target_height = int(height * (target_width / width))

print(f"Original Resolution: {width}x{height}")
print(f"New Resolution: {target_width}x{target_height}")

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # resize the frame while maintaining the aspect ratio
    resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    out.write(resized_frame)

cap.release()
out.release()
print("Video successfully resized and saved:", output_path)