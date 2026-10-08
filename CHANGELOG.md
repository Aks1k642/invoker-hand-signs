# Changelog

All notable changes to this project are documented here.

## [1.0.1] — 2026-10-08

### Fixed

- Pin Flet to 0.27.6 so the Windows executable uses the API expected by the
  Material 3 app (`ft.app`) instead of the incompatible Flet 1.x API.

## [1.0.0] — 2026-10-08

### Added

- Webcam hand-sign recognition for Quas, Wex, Exort and Invoke.
- Protection against accidental repeats using hold and release delays.
- Optional keyboard output with editable `Q/W/E/R` bindings.
- Material 3 desktop app for Windows and Linux: camera selection, preview,
  gesture status, sequence indicator and safe keyboard-output switch.
- Configurable camera and recognition timing through JSON.

### Notes

- The application starts in preview-only mode; keyboard output is opt-in.
- On Linux, keyboard output is intended for X11/Xorg. Wayland may block it.
