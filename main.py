#!/usr/bin/env python3
"""Invoker Hand Signs — webcam gesture controller for Dota 2.

Default mode is preview-only.  Add --send-keys only after calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
    import mediapipe as mp
except ImportError as error:
    missing = error.name or "a required package"
    raise SystemExit(
        f"Missing {missing}. Create a virtual environment and run: pip install -r requirements.txt"
    ) from error


DEFAULT_CONFIG: dict[str, Any] = {
    "camera_index": 0,
    "hold_seconds": 0.35,
    "release_seconds": 0.18,
    "show_landmarks": True,
    "gesture_keys": {"quas": "q", "wex": "w", "exort": "e", "invoke": "r"},
}

GESTURE_INFO = {
    "QUAS": ("Q", "QUAS · ладонь", (55, 190, 255)),
    "WEX": ("W", "WEX · кулак", (255, 150, 40)),
    "EXORT": ("E", "EXORT · V-знак", (50, 85, 255)),
    "INVOKE": ("R", "INVOKE · две ладони", (220, 70, 220)),
}


def load_config(path: Path | None) -> dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if path is None:
        return config
    try:
        supplied = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Cannot read config {path}: {error}") from error
    for key, value in supplied.items():
        if key == "gesture_keys" and isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


def finger_states(landmarks: Any) -> tuple[bool, bool, bool, bool]:
    """Return whether index, middle, ring and little fingers point upward.

    The signs are intentionally front-facing: this keeps them robust with a
    normal laptop webcam and makes them easy to learn.
    """
    pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
    return tuple(landmarks[tip].y < landmarks[pip].y for tip, pip in pairs)  # type: ignore[return-value]


def is_open_palm(landmarks: Any) -> bool:
    return sum(finger_states(landmarks)) >= 4


def classify_single_hand(landmarks: Any) -> str | None:
    index, middle, ring, little = finger_states(landmarks)
    if index and middle and ring and little:
        return "QUAS"
    if not index and not middle and not ring and not little:
        return "WEX"
    if index and middle and not ring and not little:
        return "EXORT"
    return None


def classify_hands(hand_landmarks: list[Any] | None) -> str | None:
    if not hand_landmarks:
        return None
    # Two open palms are reserved for Invoke. With any other two-hand pose we
    # intentionally emit nothing, which prevents ambiguous accidental casts.
    if len(hand_landmarks) >= 2:
        if is_open_palm(hand_landmarks[0].landmark) and is_open_palm(hand_landmarks[1].landmark):
            return "INVOKE"
        return None
    return classify_single_hand(hand_landmarks[0].landmark)


@dataclass
class GestureDebouncer:
    hold_seconds: float
    release_seconds: float
    candidate: str | None = None
    candidate_since: float = 0.0
    active: str | None = None
    last_seen: float = 0.0

    def update(self, raw: str | None, now: float) -> str | None:
        """Emit one event only after a sustained sign and a neutral release."""
        if raw is None:
            self.candidate = None
            if self.active is not None and now - self.last_seen >= self.release_seconds:
                self.active = None
            return None

        self.last_seen = now
        if self.active is not None:
            return None
        if raw != self.candidate:
            self.candidate = raw
            self.candidate_since = now
            return None
        if now - self.candidate_since >= self.hold_seconds:
            self.active = raw
            return raw
        return None


class KeySink:
    def __init__(self, enabled: bool, keys: dict[str, str]) -> None:
        self.enabled = False
        self.keys = keys
        self.controller = None
        if enabled:
            self.set_enabled(True)

    def set_enabled(self, enabled: bool) -> None:
        """Create the platform-native keyboard controller only when needed."""
        if enabled and self.controller is None:
            try:
                from pynput.keyboard import Controller
            except ImportError as error:
                raise SystemExit("Key sending requires pynput: pip install -r requirements.txt") from error
            self.controller = Controller()
        self.enabled = enabled

    def send(self, gesture: str) -> str:
        key_name = {"QUAS": "quas", "WEX": "wex", "EXORT": "exort", "INVOKE": "invoke"}[gesture]
        key = self.keys[key_name]
        if self.enabled and self.controller is not None:
            self.controller.press(key)
            self.controller.release(key)
        return key.upper()


def draw_hud(frame: Any, raw: str | None, active: str | None, sequence: list[str], sends_keys: bool) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 114), (10, 12, 22), -1)
    mode = "KEYS ON" if sends_keys else "PREVIEW — клавиши не отправляются"
    color = (65, 240, 95) if sends_keys else (70, 195, 255)
    cv2.putText(frame, mode, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    cv2.putText(frame, "ESC: выход   K: включить/выключить клавиши", (20, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    sequence_text = "  ".join(sequence[-3:]) or "—"
    cv2.putText(frame, f"ПЕЧАТИ: {sequence_text}", (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2, cv2.LINE_AA)

    shown = active or raw
    if shown:
        symbol, caption, bgr = GESTURE_INFO[shown]
        cv2.circle(frame, (w - 76, 60), 47, bgr, -1)
        cv2.putText(frame, symbol, (w - 96, 80), cv2.FONT_HERSHEY_DUPLEX, 1.8, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, caption, (w - 260, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bgr, 2, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control Invoker spheres with webcam hand signs.")
    parser.add_argument("--config", type=Path, help="Path to a JSON settings file")
    parser.add_argument("--send-keys", action="store_true", help="Enable real Q/W/E/R key injection")
    parser.add_argument("--camera", type=int, help="Override camera index from config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    camera_index = args.camera if args.camera is not None else int(config["camera_index"])
    sink = KeySink(args.send_keys, config["gesture_keys"])
    debouncer = GestureDebouncer(float(config["hold_seconds"]), float(config["release_seconds"]))

    # DirectShow avoids a slow camera probing dialog on many Windows systems.
    camera_backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    camera = cv2.VideoCapture(camera_index, camera_backend)
    if not camera.isOpened():
        raise SystemExit(f"Camera {camera_index} is unavailable. Try --camera 1.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    sequence: list[str] = []
    last_action = ""
    hands_api = mp.solutions.hands
    drawer = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles
    print("Camera started. Preview mode is safe. Press K to toggle key sending.")

    try:
        with hands_api.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        ) as hands:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Camera frame unavailable.", file=sys.stderr)
                    break
                frame = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                raw = classify_hands(result.multi_hand_landmarks)
                event = debouncer.update(raw, time.monotonic())

                if result.multi_hand_landmarks and config["show_landmarks"]:
                    for landmark_set in result.multi_hand_landmarks:
                        drawer.draw_landmarks(
                            frame, landmark_set, hands_api.HAND_CONNECTIONS,
                            styles.get_default_hand_landmarks_style(),
                            styles.get_default_hand_connections_style(),
                        )

                if event:
                    key = sink.send(event)
                    if event == "INVOKE":
                        last_action = f"Invoke → {key}"
                        sequence.clear()
                    else:
                        sequence.append(GESTURE_INFO[event][0])
                        last_action = f"{GESTURE_INFO[event][1]} → {key}"
                    print(last_action)

                draw_hud(frame, raw, debouncer.active, sequence, sink.enabled)
                if last_action:
                    cv2.putText(frame, last_action, (20, frame.shape[0] - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
                cv2.imshow("Invoker Hand Signs", frame)
                keypress = cv2.waitKey(1) & 0xFF
                if keypress in (27, ord("q")):
                    break
                if keypress in (ord("k"), ord("K")):
                    sink.set_enabled(not sink.enabled)
                    print("Key sending:", "ON" if sink.enabled else "OFF")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
