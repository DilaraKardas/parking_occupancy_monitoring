import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from core.settings import PARKING_SLOTS_PATH, ROI_PATH
from utils.roi_selector import draw_polygon, load_roi, mouse_callback, save_roi


class TestRoiSelector(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((540, 960, 3), dtype=np.uint8)
        self.sample_polygons = [
            [(100, 100), (200, 100), (200, 200), (100, 200)],
            [(300, 300), (420, 310), (380, 420)],
        ]

    def test_save_and_load_roi_roundtrip(self):
        """Saving and loading polygons should preserve exact vertex coordinates."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_slots.json"
            save_roi(self.sample_polygons, self.frame, Path("videos/sample.mp4"), out_file)

            self.assertTrue(out_file.exists())
            loaded = load_roi(self.frame, out_file)
            self.assertEqual(loaded, self.sample_polygons)

    def test_load_roi_missing_or_empty_returns_empty_list(self):
        """Loading a non-existent or 0-byte file should return an empty list."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_file = Path(tmp_dir) / "does_not_exist.json"
            self.assertEqual(load_roi(self.frame, missing_file), [])

            empty_file = Path(tmp_dir) / "empty.json"
            empty_file.write_text("", encoding="utf-8")
            self.assertEqual(load_roi(self.frame, empty_file), [])

    def test_load_roi_dimension_mismatch_raises(self):
        """Loading ROI on a frame with different dimensions should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_roi.json"
            save_roi(self.sample_polygons, self.frame, Path("videos/sample.mp4"), out_file)

            wrong_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            with self.assertRaises(ValueError):
                load_roi(wrong_frame, out_file)

    def test_load_roi_invalid_polygon_raises(self):
        """A polygon with fewer than 3 vertices should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_file = Path(tmp_dir) / "bad_roi.json"
            bad_data = {
                "video": "sample.mp4",
                "frame_width": 960,
                "frame_height": 540,
                "polygons": [[[10, 10], [20, 20]]],
            }
            bad_file.write_text(json.dumps(bad_data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_roi(self.frame, bad_file)

    def test_draw_polygon_renders_on_copy(self):
        """Drawing polygons and labels should not mutate the input frame."""
        current_points = [(50, 50), (80, 90)]
        rendered = draw_polygon(
            self.frame,
            self.sample_polygons,
            current_points,
            show_labels=True,
        )
        self.assertEqual(rendered.shape, self.frame.shape)
        self.assertEqual(int(self.frame.sum()), 0)
        self.assertGreater(int(rendered.sum()), 0)

    def test_mouse_callback_adds_point_on_left_click(self):
        """Left mouse button down event should append coordinates to current_points."""
        import cv2

        pts = []
        mouse_callback(cv2.EVENT_LBUTTONDOWN, 123, 456, 0, {"current_points": pts})
        self.assertEqual(pts, [(123, 456)])

    def test_committed_roi_and_slots_files_are_valid(self):
        """Committed parking_roi.json and parking_slots.json must load cleanly for 960x540."""
        for path in (ROI_PATH, PARKING_SLOTS_PATH):
            self.assertTrue(path.exists(), f"Expected ROI file missing: {path}")
            polygons = load_roi(self.frame, path)
            self.assertGreater(len(polygons), 0)
            for poly in polygons:
                self.assertGreaterEqual(len(poly), 3)
                for x, y in poly:
                    self.assertTrue(0 <= x <= 960, f"x={x} out of bounds in {path.name}")
                    self.assertTrue(0 <= y <= 540, f"y={y} out of bounds in {path.name}")


if __name__ == "__main__":
    unittest.main()
