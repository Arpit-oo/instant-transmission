"""
Instant Transmission - vanish from your webcam feed with Goku's two-finger gesture.

Pipeline: webcam -> gesture + person segmentation -> composite over clean plate
          -> OBS Virtual Camera (pick it in Zoom/Meet/Teams) + local preview.

Keys (preview window focused):
  SPACE  vanish / reappear manually      B  recapture background (step out)
  T      toggle teleport mode            M  toggle sound
  D      toggle debug overlay            Q / ESC  quit
"""
import argparse
import time

import cv2
import numpy as np

from effects import FX
from gesture import GestureDetector
from sfx import Sfx
from vision import Camera, CleanPlate, Segmenter

VISIBLE, VANISHING, GONE, APPEARING = "VISIBLE", "VANISHING", "GONE", "APPEARING"
PROC_W = 640  # ML runs at this width


def parse_args():
    p = argparse.ArgumentParser(description="Instant Transmission virtual camera")
    p.add_argument("--camera", type=int, default=0, help="webcam index")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--hold", type=float, default=0.3, help="seconds to hold gesture")
    p.add_argument("--vanish-time", type=float, default=0.35)
    p.add_argument("--appear-time", type=float, default=0.35)
    p.add_argument("--ghost-time", type=float, default=0.6)
    p.add_argument("--no-vcam", action="store_true", help="preview only")
    p.add_argument("--no-capture", action="store_true", help="skip startup background capture")
    p.add_argument("--ghost", action="store_true", help="fading afterimage after vanishing")
    p.add_argument("--glow", action="store_true", help="cyan rim glow during transitions")
    p.add_argument("--teleport", action="store_true", help="reappear somewhere else in frame")
    p.add_argument("--mirror-output", action="store_true", help="mirror the virtual cam feed")
    p.add_argument("--mute", action="store_true", help="no sound effects")
    return p.parse_args()


def open_vcam(w, h, fps):
    try:
        import pyvirtualcam

        cam = pyvirtualcam.Camera(width=w, height=h, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR)
        print(f"[vcam] streaming to: {cam.device}")
        return cam
    except Exception as e:
        print(f"[vcam] unavailable ({e}). Install OBS Studio for the virtual camera driver. Preview only.")
        return None


def pick_teleport_dx(soft, w, rng):
    cols = np.where(soft.max(axis=0) > 0.5)[0]
    if cols.size == 0:
        return 0
    lo, hi = -int(cols[0]), int(w - 1 - cols[-1])
    far = [dx for dx in range(lo, hi + 1, 8) if abs(dx) > 0.15 * w]
    return int(rng.choice(far)) if far else 0


def draw_hud(img, state, progress, fps, plate, vcam_on, teleport, sound, status, gest, debug, scale):
    h, w = img.shape[:2]
    colors = {VISIBLE: (120, 220, 120), VANISHING: (255, 220, 120), GONE: (255, 180, 60), APPEARING: (255, 220, 120)}
    cv2.rectangle(img, (0, 0), (w, 34), (20, 20, 20), -1)
    line = f"{state}  |  {fps:4.1f} fps  |  vcam {'ON' if vcam_on else 'OFF'}  |  teleport {'ON' if teleport else 'off'}  |  sound {'on' if sound else 'off'}  |  bg {plate.coverage * 100:3.0f}%"
    cv2.putText(img, line, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors.get(state, (255, 255, 255)), 1, cv2.LINE_AA)
    if progress > 0:
        cv2.rectangle(img, (10, h - 22), (10 + int((w - 20) * progress), h - 10), (255, 235, 140), -1)
        cv2.rectangle(img, (10, h - 22), (w - 10, h - 10), (255, 235, 140), 1)
    if status:
        (tw, th), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
        cv2.putText(img, status, ((w - tw) // 2, h // 2), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 235, 140), 2, cv2.LINE_AA)
    cv2.putText(img, "SPACE vanish  B background  T teleport  M sound  D debug  Q quit",
                (10, h - 30 if progress > 0 else h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    if debug:
        # gesture landmarks are in processing-frame px; preview is mirrored
        def P(pt):
            return (int(w - pt[0] * scale), int(pt[1] * scale))

        face = gest.debug.get("face")
        if face:
            cv2.line(img, P(face[0]), P(face[1]), (0, 200, 255), 3)
        y = 60
        for hand in gest.debug.get("hands", []):
            for tip in hand["tips"]:
                cv2.circle(img, P(tip), 7, (0, 255, 0) if hand["on_forehead"] else (0, 0, 255), -1)
            flags = "  ".join(f"{k}:{'Y' if hand[k] else 'n'}" for k in ("two_up", "others_down", "together", "on_forehead"))
            cv2.putText(img, flags, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22


def main():
    args = parse_args()
    cam = Camera(args.camera, args.width, args.height, args.fps)
    w, h = cam.size
    proc_size = (PROC_W, int(round(h * PROC_W / w)))
    scale_proc_to_full = w / PROC_W
    print(f"[cam] {w}x{h}")

    seg = Segmenter((w, h))
    gest = GestureDetector(hold_s=args.hold)
    plate = CleanPlate((w, h))
    sfx = Sfx(enabled=not args.mute)
    vcam = None if args.no_vcam else open_vcam(w, h, args.fps)
    fx = FX((w, h), glow=args.glow)
    rng = np.random.default_rng()

    state, t_state = VISIBLE, time.time()
    teleport, sound, debug = args.teleport, not args.mute, False
    dx = 0
    ghost = None  # ((tint, alpha), t0)
    fps, t_prev = 0.0, time.time()
    if not args.no_capture:
        plate.start_capture(time.time())

    preview_w = min(w, 1100)
    preview_scale = preview_w / w
    win = "Instant Transmission"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def go(new_state):
        nonlocal state, t_state
        state, t_state = new_state, time.time()

    def toggle(soft):
        nonlocal dx
        if state == VISIBLE:
            plate.fill_unknown()
            go(VANISHING)
            if sound:
                sfx.play(sfx.vanish)
        elif state == GONE:
            if teleport:
                dx = pick_teleport_dx(soft, w, rng)
            go(APPEARING)
            if sound:
                sfx.play(sfx.appear)

    try:
        while True:
            frame = cam.read()
            now = time.time()
            dt = now - t_prev
            t_prev = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0)

            small = cv2.resize(frame, proc_size, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            soft, removal, learn_ok = seg.process(rgb)

            status = None
            progress = 0.0
            out = frame

            if plate.capturing:
                status = plate.tick_capture(frame, now)
            else:
                plate.update(frame, learn_ok)
                fired, progress = gest.update(rgb, now)

                if fired:
                    toggle(soft)

                # ---- render ----
                t = max(0.0, now - t_state)
                if state == VISIBLE and dx == 0 and ghost is None:
                    out = frame
                else:
                    base = fx.remove_person(frame, plate.matched(frame, learn_ok, removal), removal)
                    layer, alpha = fx.shift(frame, soft, dx)

                    if state == VANISHING:
                        u = min(1.0, t / args.vanish_time)
                        if ghost is None and args.ghost:
                            ghost = (fx.make_ghost(layer, alpha), now)
                        out = fx.transition(base, layer, alpha, u)
                        if u >= 1.0:
                            go(GONE)
                            dx = 0
                    elif state == APPEARING:
                        u = min(1.0, t / args.appear_time)
                        out = fx.transition(base, layer, alpha, u, appearing=True)
                        if u >= 1.0:
                            go(VISIBLE)
                    elif state == GONE:
                        out = base
                    else:  # VISIBLE: teleported, or ghost still fading
                        out = fx.over(base, layer, alpha) if dx else frame

                    if ghost is not None:
                        gu = (now - ghost[1]) / args.ghost_time
                        if gu >= 1.0:
                            ghost = None
                        elif state in (VANISHING, GONE):
                            out = fx.draw_ghost(out, ghost[0], gu)

            # ---- outputs ----
            if vcam is not None:
                vframe = cv2.flip(out, 1) if args.mirror_output else out
                vcam.send(np.ascontiguousarray(vframe))

            preview = cv2.flip(out, 1)
            if preview_scale != 1:
                preview = cv2.resize(preview, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA)
            draw_hud(preview, state, progress, fps, plate, vcam is not None, teleport, sound,
                     status, gest, debug, scale_proc_to_full * preview_scale)
            cv2.imshow(win, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord(" ") and not plate.capturing:
                toggle(soft)
            elif key == ord("b"):
                plate.start_capture(now)
            elif key == ord("t"):
                teleport = not teleport
            elif key == ord("m"):
                sound = not sound
            elif key == ord("d"):
                debug = not debug
    finally:
        cam.close()
        seg.close()
        gest.close()
        if vcam is not None:
            vcam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
