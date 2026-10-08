#!/usr/bin/env python3
"""Material 3 desktop UI for Invoker Hand Signs.

Run ``python app.py`` after installing requirements.txt.
"""

from __future__ import annotations

import asyncio
import base64
import sys
import time
from typing import Any

import cv2
import flet as ft
import mediapipe as mp

from main import GESTURE_INFO, GestureDebouncer, KeySink, classify_hands


def camera_backend() -> int:
    return cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY


def detect_cameras(limit: int = 5) -> list[int]:
    """Find cameras without keeping any device handle open."""
    found: list[int] = []
    for index in range(limit):
        camera = cv2.VideoCapture(index, camera_backend())
        if camera.isOpened():
            found.append(index)
        camera.release()
    return found


def action_label(gesture: str) -> str:
    symbol, name, _ = GESTURE_INFO[gesture]
    return f"{symbol} · {name.split(' · ')[0]}"


async def main(page: ft.Page) -> None:
    page.title = "Invoker Hand Signs"
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = ft.Theme(color_scheme_seed=ft.Colors.DEEP_PURPLE)
    page.dark_theme = ft.Theme(color_scheme_seed=ft.Colors.DEEP_PURPLE)
    page.padding = 0
    page.window.min_width = 980
    page.window.min_height = 720

    available_cameras = detect_cameras()
    camera_select = ft.Dropdown(
        label="Камера",
        value=str(available_cameras[0]) if available_cameras else None,
        options=[ft.dropdown.Option(str(index), f"Камера {index}") for index in available_cameras],
        expand=True,
        disabled=not available_cameras,
    )
    preview = ft.Image(
        width=720,
        height=405,
        fit=ft.ImageFit.CONTAIN,
        border_radius=16,
        visible=False,
    )
    preview_empty = ft.Container(
        width=720,
        height=405,
        border_radius=16,
        bgcolor=ft.Colors.SURFACE_VARIANT,
        alignment=ft.alignment.center,
        content=ft.Column(
            [
                ft.Icon(ft.Icons.VIDEOCAM_OUTLINED, size=52),
                ft.Text("Предпросмотр выключен", size=18),
                ft.Text("Выберите камеру и нажмите «Запустить»", color=ft.Colors.ON_SURFACE_VARIANT),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=8,
        ),
    )
    current_gesture = ft.Text("Ожидание жеста", size=22, weight=ft.FontWeight.W_600)
    sequence_value = ft.Text("—", size=28, weight=ft.FontWeight.BOLD)
    status = ft.Text("Предпросмотр не запущен", color=ft.Colors.ON_SURFACE_VARIANT)
    key_switch = ft.Switch(label="Отправлять клавиши в активное окно", value=False)
    start_button = ft.FilledButton("Запустить предпросмотр", icon=ft.Icons.PLAY_ARROW)
    stop_button = ft.OutlinedButton("Остановить", icon=ft.Icons.STOP, disabled=True)

    key_fields: dict[str, ft.TextField] = {}
    for sphere, default in (("quas", "Q"), ("wex", "W"), ("exort", "E"), ("invoke", "R")):
        key_fields[sphere] = ft.TextField(
            label=sphere.capitalize(), value=default, max_length=1, width=110,
            text_align=ft.TextAlign.CENTER,
        )

    running = False
    task_started = False
    key_sink = KeySink(False, {name: field.value.lower() for name, field in key_fields.items()})

    def set_running_ui(is_running: bool, message: str) -> None:
        start_button.disabled = is_running
        stop_button.disabled = not is_running
        camera_select.disabled = is_running
        preview.visible = is_running
        preview_empty.visible = not is_running
        status.value = message
        page.update()

    def update_key_map(_: ft.ControlEvent | None = None) -> None:
        for name, field in key_fields.items():
            value = (field.value or "").strip().lower()
            if len(value) != 1:
                field.error_text = "1 символ"
            else:
                field.error_text = None
                key_sink.keys[name] = value
        page.update()

    for field in key_fields.values():
        field.on_change = update_key_map

    def toggle_keys(event: ft.ControlEvent) -> None:
        try:
            update_key_map()
            key_sink.set_enabled(bool(event.control.value))
            status.value = "Отправка клавиш включена" if key_sink.enabled else "Безопасный режим: клавиши не отправляются"
        except SystemExit as error:
            key_switch.value = False
            status.value = str(error)
        page.update()

    key_switch.on_change = toggle_keys

    async def preview_loop() -> None:
        nonlocal running, task_started
        camera = None
        try:
            selected = camera_select.value
            if selected is None:
                running = False
                set_running_ui(False, "Камера не выбрана")
                return
            camera = cv2.VideoCapture(int(selected), camera_backend())
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            if not camera.isOpened():
                running = False
                set_running_ui(False, "Не удалось открыть камеру")
                return

            sequence: list[str] = []
            debouncer = GestureDebouncer(hold_seconds=0.35, release_seconds=0.18)
            hands_api = mp.solutions.hands
            drawer = mp.solutions.drawing_utils
            styles = mp.solutions.drawing_styles
            with hands_api.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
            ) as hands:
                while running:
                    ok, frame = camera.read()
                    if not ok:
                        status.value = "Камера перестала отдавать кадры"
                        page.update()
                        break
                    frame = cv2.flip(frame, 1)
                    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    raw = classify_hands(result.multi_hand_landmarks)
                    event = debouncer.update(raw, time.monotonic())

                    if result.multi_hand_landmarks:
                        for landmark_set in result.multi_hand_landmarks:
                            drawer.draw_landmarks(
                                frame, landmark_set, hands_api.HAND_CONNECTIONS,
                                styles.get_default_hand_landmarks_style(),
                                styles.get_default_hand_connections_style(),
                            )
                    if event:
                        sent_key = key_sink.send(event)
                        if event == "INVOKE":
                            sequence.clear()
                            status.value = f"Invoke → {sent_key}"
                        else:
                            sequence.append(GESTURE_INFO[event][0])
                            status.value = f"{action_label(event)} → {sent_key}"
                    if raw:
                        current_gesture.value = action_label(raw)
                    else:
                        current_gesture.value = "Ожидание жеста"
                    sequence_value.value = "  ".join(sequence[-3:]) or "—"

                    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        preview.src_base64 = base64.b64encode(encoded).decode("ascii")
                    page.update()
                    await asyncio.sleep(0.01)
        except Exception as error:  # show a usable message instead of silently closing the GUI
            status.value = f"Ошибка камеры: {error}"
            page.update()
        finally:
            if camera is not None:
                camera.release()
            running = False
            task_started = False
            set_running_ui(False, status.value if status.value else "Предпросмотр остановлен")

    def start_preview(_: ft.ControlEvent) -> None:
        nonlocal running, task_started
        if not available_cameras:
            status.value = "Камеры не найдены. Подключите камеру и перезапустите приложение."
            page.update()
            return
        if task_started:
            return
        running = True
        task_started = True
        set_running_ui(True, "Ищу руки…")
        page.run_task(preview_loop)

    def stop_preview(_: ft.ControlEvent) -> None:
        nonlocal running
        running = False
        status.value = "Останавливаю камеру…"
        page.update()

    start_button.on_click = start_preview
    stop_button.on_click = stop_preview

    settings_card = ft.Card(
        content=ft.Container(
            padding=20,
            width=330,
            content=ft.Column(
                [
                    ft.Text("Настройки", size=22, weight=ft.FontWeight.BOLD),
                    camera_select,
                    ft.Divider(),
                    ft.Text("Бинды Invoker", weight=ft.FontWeight.W_600),
                    ft.Row(list(key_fields.values()), wrap=True),
                    ft.Text("Жесты остаются: ладонь / кулак / V / две ладони.", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
                    ft.Divider(),
                    key_switch,
                    ft.Text("Сначала проверьте жесты без отправки клавиш. Затем активируйте окно Dota 2.", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
                ],
                spacing=15,
            ),
        )
    )
    preview_card = ft.Card(
        content=ft.Container(
            padding=20,
            content=ft.Column(
                [
                    ft.Text("Предпросмотр", size=22, weight=ft.FontWeight.BOLD),
                    ft.Stack([preview_empty, preview]),
                    ft.Row([start_button, stop_button]),
                    ft.Divider(),
                    ft.Text("Текущая печать", color=ft.Colors.ON_SURFACE_VARIANT),
                    current_gesture,
                    ft.Text("Собранная последовательность", color=ft.Colors.ON_SURFACE_VARIANT),
                    sequence_value,
                    status,
                ],
                spacing=12,
            ),
        )
    )

    page.add(
        ft.AppBar(title=ft.Text("Invoker Hand Signs"), center_title=False),
        ft.Container(
            padding=24,
            content=ft.ResponsiveRow(
                [
                    ft.Column([preview_card], col={"sm": 12, "lg": 8}),
                    ft.Column([settings_card], col={"sm": 12, "lg": 4}),
                ],
                spacing=20,
                run_spacing=20,
            ),
        ),
    )


if __name__ == "__main__":
    ft.app(target=main)
