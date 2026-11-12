import json
from pathlib import Path
import cv2
import numpy as np


def mouse_callback(event, x, y, flags, param):
    """Handles mouse events for ROI polygon selection."""
    current_points = param["current_points"]

    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        print(f"Point added: ({x}, {y})")


def draw_polygon(frame, polygons, current_points, show_labels=False):
    """
    Draw completed polygons and the currently selected polygon.
    Optionally display slot IDs for completed polygons.
    """
    display_frame = frame.copy()

    # Draw completed polygons
    for i, polygon in enumerate(polygons):
        polygon_array = np.array(
            polygon,
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        # Draw polygon boundary
        cv2.polylines(
            display_frame,
            [polygon_array],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )

        # Draw polygon vertices
        for point in polygon:
            cv2.circle(
                display_frame,
                tuple(point),
                5,
                (0, 255, 0),
                -1,
            )

        # Draw slot ID at polygon center
        if show_labels:
            moments = cv2.moments(polygon_array)

            if moments["m00"] != 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])

                cv2.putText(
                    display_frame,
                    f"Slot {i + 1}",
                    (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

    # Draw the current polygon
    for point in current_points:
        cv2.circle(
            display_frame,
            tuple(point),
            5,
            (0, 0, 255),
            -1,
        )

    # Connect current polygon points
    if len(current_points) > 1:
        for i in range(len(current_points) - 1):
            cv2.line(
                display_frame,
                tuple(current_points[i]),
                tuple(current_points[i + 1]),
                (0, 0, 255),
                2,
            )

    return display_frame


def save_roi(polygons, frame, video_path, output_path):
    """Saves all ROI polygons as JSON."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    height, width = frame.shape[:2]

    roi_data = {
        "video": video_path.name,
        "frame_width": width,
        "frame_height": height,
        "polygons": [
            [list(point) for point in polygon]
            for polygon in polygons
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(roi_data, file, indent=4)

    print(f"ROI saved successfully: {output_path}")


def load_roi(frame, output_path):
    """Load previously saved ROI polygons from a JSON file."""
    output_path = Path(output_path)

    # Start with an empty ROI if the file does not exist or is empty.
    if not output_path.exists() or output_path.stat().st_size == 0:
        print("No existing ROI data found. Starting with an empty ROI.")
        return []

    with open(output_path, "r", encoding="utf-8") as file:
        roi_data = json.load(file)

    required_keys = {"frame_width", "frame_height", "polygons"}
    missing_keys = required_keys - set(roi_data.keys())
    if missing_keys:
        raise ValueError(f"ROI file is missing required keys: {sorted(missing_keys)}")

    height, width = frame.shape[:2]

    if (
        roi_data["frame_width"] != width
        or roi_data["frame_height"] != height
    ):
        raise ValueError(
            "Saved ROI dimensions do not match the current video frame. "
            f"Saved: {roi_data['frame_width']}x{roi_data['frame_height']}, "
            f"Current: {width}x{height}"
        )

    polygons = []
    for index, polygon in enumerate(roi_data["polygons"], start=1):
        if len(polygon) < 3:
            raise ValueError(f"Polygon {index} in {output_path} has fewer than 3 points.")
        polygons.append([tuple(map(int, point)) for point in polygon])

    print(f"{len(polygons)} ROI polygon(s) loaded successfully.")

    return polygons


def select_roi(video_path, output_path, show_labels=False):
    """Interactive OpenCV window for drawing and saving ROI polygons."""
    video_path = Path(video_path)
    output_path = Path(output_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError("Could not read the first video frame.")

    window_name = "ROI Selection"

    # Stores completed polygons
    polygons = load_roi(frame, output_path)

    # Stores points of the polygon currently being drawn
    current_points = []

    callback_params = {"current_points": current_points}

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(
        window_name,
        mouse_callback,
        callback_params,
    )

    print("\nROI Selection Controls")
    print("----------------------")
    print("Left Click : Add point")
    print("Backspace  : Undo last point")
    print("R          : Clear current polygon")
    print("D          : Delete last completed polygon")
    print("ENTER      : Close current polygon")
    print("S          : Save all polygons")
    print("ESC        : Cancel selection")

    while True:
        display_frame = draw_polygon(
            frame,
            polygons,
            current_points,
            show_labels=show_labels,
        )

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(20) & 0xFF

        # Undo last point
        if key in (8, 127):
            if current_points:
                removed_point = current_points.pop()
                print(f"Point removed: {removed_point}")

        # Clear current polygon
        elif key in (ord("r"), ord("R")):
            current_points.clear()
            print("Current polygon cleared.")

        # Delete last completed polygon
        elif key in (ord("d"), ord("D")):
            if polygons:
                polygons.pop()
                print(f"Removed last completed polygon. Remaining: {len(polygons)}")
            else:
                print("No completed polygons to remove.")

        # Close current polygon
        elif key in (10, 13):
            if len(current_points) >= 3:
                polygons.append(current_points.copy())
                current_points.clear()
                print(f"Polygon {len(polygons)} completed.")
            else:
                print("A polygon needs at least 3 points.")

        # Save all polygons
        elif key in (ord("s"), ord("S")):
            if current_points:
                print("Close the current polygon with ENTER first.")
                continue

            if not polygons:
                print("No completed polygons to save.")
                continue

            save_roi(
                polygons,
                frame,
                video_path,
                output_path,
            )
            break

        # Cancel selection
        elif key == 27:
            print("ROI selection cancelled.")
            break

    cv2.destroyAllWindows()
