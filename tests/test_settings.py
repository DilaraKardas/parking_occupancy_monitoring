from pathlib import Path
import pytest
from core.settings import (
    PROJECT_ROOT,
    CONFIG_PATH,
    VIDEO_PATH,
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    DEVICE,
)

def test_project_structure_and_paths():
    """Proje kök dizininin ve config dosyasının var olduğunu doğrular."""
    assert isinstance(PROJECT_ROOT, Path)
    assert PROJECT_ROOT.exists(), f"Proje kök dizini bulunamadı: {PROJECT_ROOT}"
    
    assert isinstance(CONFIG_PATH, Path)
    assert CONFIG_PATH.exists(), f"config.yaml dosyası bulunamadı: {CONFIG_PATH}"

def test_settings_loaded_correctly():
    """Config dosyasından gelen kritik ayar değişkenlerinin tiplerini ve değerlerini kontrol eder."""
    # Eşik değerlerinin mantıklı aralıkta olduğunu test et
    assert isinstance(CONFIDENCE_THRESHOLD, float)
    assert 0.0 <= CONFIDENCE_THRESHOLD <= 1.0

    # Cihaz seçiminin (Device) string olduğunu kontrol et
    assert isinstance(DEVICE, str)
    assert DEVICE in ["cpu", "cuda:0"] or DEVICE.startswith("cuda:")

def test_model_and_video_paths_configured():
    """Model ve video yollarının Path nesnesi olduğunu doğrular."""
    assert isinstance(VIDEO_PATH, Path)
    assert isinstance(MODEL_PATH, Path)