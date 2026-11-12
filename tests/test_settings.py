import importlib
from pathlib import Path
import unittest

from core.settings import (
    CONFIDENCE_THRESHOLD,
    CONFIG_PATH,
    DEVICE,
    MODEL_PATH,
    PARKING_SLOTS_PATH,
    PROJECT_ROOT,
    ROI_PATH,
    VIDEO_PATH,
)


class TestSettings(unittest.TestCase):
    def test_project_structure_and_paths(self):
        """Verify that the project root and config.yaml exist."""
        self.assertIsInstance(PROJECT_ROOT, Path)
        self.assertTrue(PROJECT_ROOT.exists(), f"Project root directory not found: {PROJECT_ROOT}")

        self.assertIsInstance(CONFIG_PATH, Path)
        self.assertTrue(CONFIG_PATH.exists(), f"config.yaml file not found: {CONFIG_PATH}")

    def test_settings_loaded_correctly(self):
        """Verify that critical setting variables have valid types and ranges."""
        self.assertIsInstance(CONFIDENCE_THRESHOLD, float)
        self.assertTrue(0.0 <= CONFIDENCE_THRESHOLD <= 1.0)

        self.assertIsInstance(DEVICE, str)
        self.assertTrue(DEVICE in ["cpu", "cuda:0"] or DEVICE.startswith("cuda:"))

    def test_model_video_and_roi_paths_configured(self):
        """Verify that model, video, and ROI paths are configured as Path objects."""
        for path_obj in (VIDEO_PATH, MODEL_PATH, ROI_PATH, PARKING_SLOTS_PATH):
            self.assertIsInstance(path_obj, Path)
            self.assertTrue(path_obj.is_absolute())

    def test_roi_entrypoints_import_cleanly(self):
        """Ensure ROI selector entrypoint modules can be imported without error."""
        select_roi_mod = importlib.import_module("roi.select_roi")
        select_slots_mod = importlib.import_module("roi.select_slots")
        self.assertTrue(hasattr(select_roi_mod, "ROI_PATH"))
        self.assertTrue(hasattr(select_slots_mod, "PARKING_SLOTS_PATH"))


if __name__ == "__main__":
    unittest.main()
