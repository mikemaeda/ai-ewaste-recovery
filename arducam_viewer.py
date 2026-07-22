"""Open a plain live Arducam/webcam window without YOLO.

Use this to verify camera access before running the AI dashboard.
"""

from __future__ import annotations

import argparse

import cv2

from ewaste_research.camera import open_camera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plain live camera viewer.")
    parser.add_argument(
        "--source",
        default="auto",
        help='Camera index, video path, or "auto" to scan indexes 0-9.',
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "default", "dshow", "msmf"],
        default="auto",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture, info = open_camera(args.source, args.backend, args.width, args.height)

    window = f"Arducam viewer - source {info['source']} ({info['backend']})"
    print(
        f"Opened camera source {info['source']} with backend {info['backend']} "
        f"at {info.get('width')}x{info.get('height')} fps={info.get('fps'):.1f}"
    )
    print("Camera viewer open. Press Q or Esc in the window to quit.")
    while True:
        ok, frame = capture.read()
        if not ok:
            print("No frame received from camera.")
            break

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
