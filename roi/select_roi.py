from core.settings import VIDEO_PATH, ROI_PATH
from utils.roi_selector import select_roi


if __name__ == "__main__":
    select_roi(
        VIDEO_PATH,
        ROI_PATH
    )