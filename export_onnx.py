from ultralytics import YOLO

# Mevcut PyTorch modelini yükle
model = YOLO("models/yolo26s.pt")

# ONNX formatına export et (Eğer imgsz'yi 960 yapacaksan aşağıdaki değeri 960 olarak güncelleyebilirsin)
model.export(format="onnx", imgsz=640, simplify=True)