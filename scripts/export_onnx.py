from ultralytics import YOLO
from core.settings import PROJECT_ROOT

# Load the YOLO model
model_path = PROJECT_ROOT / "models" / "yolo26s.pt"
model = YOLO(str(model_path))
# convert the model to ONNX format
model.export(format="onnx", imgsz=640, simplify=True)