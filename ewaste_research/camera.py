"""Camera discovery helpers for OpenCV on Windows/USB cameras."""

from __future__ import annotations

import cv2

CAMERA_BACKENDS = {
    "default": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}

BACKEND_ORDER = ["default", "msmf", "dshow"]


def open_camera(
    source: str,
    backend: str,
    width: int,
    height: int,
    max_index: int = 9,
) -> tuple[cv2.VideoCapture, dict]:
    """
    Open a camera or video source and return its capture plus selected settings.

    source can be a numeric camera index, "auto", or a video path/URL.
    backend can be "default", "msmf", "dshow", or "auto".
    """
    if not source.isdigit() and source != "auto":
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video file/source: {source}")
        return capture, {"source": source, "backend": "file"}

    indexes = range(max_index + 1) if source == "auto" else [int(source)]
    backends = BACKEND_ORDER if backend == "auto" else [backend]
    attempted: list[str] = []

    for backend_name in backends:
        backend_id = CAMERA_BACKENDS[backend_name]
        for index in indexes:
            attempted.append(f"{index}/{backend_name}")
            capture = cv2.VideoCapture(index, backend_id)
            if not capture.isOpened():
                capture.release()
                continue

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            ok = False
            for _ in range(10):
                ok, frame = capture.read()
                if ok and frame is not None:
                    break
            if ok:
                actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                return capture, {
                    "source": index,
                    "backend": backend_name,
                    "width": actual_width,
                    "height": actual_height,
                    "fps": fps,
                }
            capture.release()

    raise RuntimeError(
        "Could not open any camera. Tried: "
        + ", ".join(attempted)
        + ". Close Camera/Zoom/Teams/Discord/browser tabs, check Windows camera "
        "privacy settings, then retry."
    )

