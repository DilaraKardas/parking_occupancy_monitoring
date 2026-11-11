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
    #Test that the project structure and paths are set up correctly.
    assert isinstance(PROJECT_ROOT, Path)
    assert PROJECT_ROOT.exists(), f"Project root directory not found: {PROJECT_ROOT}"

    assert isinstance(CONFIG_PATH, Path)
    assert CONFIG_PATH.exists(), f"config.yaml file not found: {CONFIG_PATH}"

def test_settings_loaded_correctly():
    #Test that the critical setting variables loaded from the config file have the correct types and values.
    # Test that the threshold values are in a logical range
    assert isinstance(CONFIDENCE_THRESHOLD, float)
    assert 0.0 <= CONFIDENCE_THRESHOLD <= 1.0

    # Test that the device selection is a string
    assert isinstance(DEVICE, str)
    assert DEVICE in ["cpu", "cuda:0"] or DEVICE.startswith("cuda:")

def test_model_and_video_paths_configured():
    #Test that the model and video paths are configured as Path objects.
    assert isinstance(VIDEO_PATH, Path)
    assert isinstance(MODEL_PATH, Path)