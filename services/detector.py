from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from core.settings import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IMGSZ,
    DEVICE,
    MODEL_IOU,
    VEHICLE_CLASSES,
    SLICE_HEIGHT,
    SLICE_WIDTH,
    OVERLAP_HEIGHT_RATIO,
    OVERLAP_WIDTH_RATIO,
    POSTPROCESS_TYPE,
    POSTPROCESS_MATCH_METRIC,
    POSTPROCESS_MATCH_THRESHOLD,
)

def load_model():
    """YOLO ve SAHI tabanlı algılama modelini yükler."""
    print(f"Loading model: {MODEL_PATH}")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        image_size=IMGSZ,
        device=DEVICE,
    )
    # Model yalnızca araç sınıflarını tahmin etsin.
    detection_model.model.overrides["classes"] = list(VEHICLE_CLASSES)
    
    # Ultralytics NMS IoU eşiğini uygula
    detection_model.model.overrides["iou"] = MODEL_IOU
    return detection_model

def run_inference(frame_rgb, detection_model):
    """Verilen RGB kare üzerinde dilimlenmiş (sliced) tahmin çalıştırır."""
    result = get_sliced_prediction(
        image=frame_rgb,
        detection_model=detection_model,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=OVERLAP_WIDTH_RATIO,
        perform_standard_pred=False,
        postprocess_type=POSTPROCESS_TYPE,
        postprocess_match_metric=POSTPROCESS_MATCH_METRIC,
        postprocess_match_threshold=POSTPROCESS_MATCH_THRESHOLD,
        verbose=0,
        progress_bar=False,
    )
    return result