# Instant Transmission

Put two fingers (index + middle) on your forehead, hold ~0.3s → you vanish from your webcam feed. Do it again → you reappear.

## Setup (once)
1. Install OBS Studio (provides the virtual camera driver): `winget install OBSProject.OBSStudio`
2. Close any app using the webcam.
3. `run.bat` (creates venv on first run).

In Zoom / Meet / Teams, choose **OBS Virtual Camera** as the camera.

## Usage
On start: **step fully out of frame**. It waits until it sees an empty room, then captures it. Can't leave your seat? Press SPACE to skip: it learns the room as you move and fills unseen spots from their surroundings (never from frames with you in them).

| Key | Action |
|---|---|
| SPACE | vanish / reappear manually (skips background capture while capturing) |
| B | recapture background (waits for empty room) |
| T | teleport mode (reappear elsewhere in frame) |
| M | sound on/off |
| D | debug overlay (gesture checks) |
| Q / ESC | quit |

Flags: `--camera 1`, `--hold 0.5`, `--teleport`, `--ghost` (fading afterimage), `--glow` (cyan rim glow), `--mute`, `--no-vcam`, `--mirror-output`, `--width/--height/--fps`.

## Google Meet
1. Start `run.bat` **before** joining (keep it running; Meet shows black if it closes).
2. Meet → ⋮ → Settings → Video → Camera → **OBS Virtual Camera** (refresh the tab if it's missing).
3. Turn off Meet's background effects, lighting adjustment, and framing.

## How it works
- **Gesture**: MediaPipe Hands + Face Mesh. Index + middle extended and together, ring + pinky curled, fingertips on the forehead line, held 0.3s.
- **Vanish**: MediaPipe Selfie Segmentation erases you and fills the hole from a clean background plate. The erase mask is the union of recent frames, pushed forward along your motion (segmentation lags real movement), dilated and feathered outward, so no edges or fragments leak through when you move.
- **Background plate**: captured only once the room is verifiably empty. Afterwards, pixels are refreshed only when far from you **and** already matching the plate, so a person can never be absorbed into it.
- **Invisible, not a silhouette**: every frame the plate is re-matched to the live camera: a local per-channel gain map (webcam auto-exposure / white balance drift) plus fresh sensor grain at the camera's measured noise level. Without this, the erased area reads as a slightly-off, too-clean outline.
- **Output**: `pyvirtualcam` → OBS Virtual Camera.

Tips: steady lighting, keep the camera still (the background plate assumes it), press B if lighting changes.
