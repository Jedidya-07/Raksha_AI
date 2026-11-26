import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from detector.image_detector import DEVICE, TRANSFORM

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _crop_primary_face(rgb_frame: np.ndarray) -> tuple[Image.Image, bool]:
    """Detects the largest face in the frame and crops it with a slight margin."""
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

    if len(faces) == 0:
        return Image.fromarray(rgb_frame), False

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad_x = int(w * 0.15)
    pad_y = int(h * 0.15)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(rgb_frame.shape[1], x + w + pad_x)
    y2 = min(rgb_frame.shape[0], y + h + pad_y)
    return Image.fromarray(rgb_frame[y1:y2, x1:x2]), True


def _save_frame_evidence(top_frames: list[dict], out_path: str) -> None:
    """Saves up to 2 highest-risk sampled frames side-by-side with timestamp labels."""
    tile_w, tile_h = 320, 240
    count = max(1, len(top_frames))
    canvas = Image.new("RGB", (tile_w * count, tile_h + 36), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)

    for idx, item in enumerate(top_frames):
        thumb = item["pil_frame"].resize((tile_w, tile_h))
        x_offset = idx * tile_w
        canvas.paste(thumb, (x_offset, 0))
        label = f"t={item['timestamp']:.1f}s | Risk: {item['score']*100:.1f}%"
        draw.text((x_offset + 10, tile_h + 10), label, fill=(248, 250, 252))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


def analyze_video(
    filepath: str,
    model,
    fake_idx: int,
    evidence_out_path: str,
    loaded_custom: bool = True,
    num_samples: int = 8,
) -> dict:
    """Samples frames across a video, scores faces/frames, and outputs suspicious timestamps."""
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise ValueError("Unable to open video file for analysis.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("Video appears empty or corrupted.")

    sample_indices = np.linspace(0, max(0, total_frames - 1), num=min(num_samples, total_frames), dtype=int)
    frame_results = []
    faces_found = 0

    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        crop_pil, has_face = _crop_primary_face(rgb)
        if has_face:
            faces_found += 1

        tensor = TRANSFORM(crop_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]
            fake_prob = float(probs[fake_idx].item())

        frame_results.append({
            "frame_idx": int(frame_idx),
            "timestamp": round(float(frame_idx / fps), 2),
            "score": fake_prob,
            "has_face": has_face,
            "pil_frame": Image.fromarray(rgb),
        })

    cap.release()

    if not frame_results:
        raise ValueError("No readable frames could be extracted from the video.")

    # Sort by highest manipulation score
    ranked = sorted(frame_results, key=lambda r: r["score"], reverse=True)
    top_k = ranked[: min(3, len(ranked))]
    agg_score = float(np.mean([r["score"] for r in top_k]))
    score_std = float(np.std([r["score"] for r in frame_results]))

    _save_frame_evidence(ranked[:2], evidence_out_path)

    flagged_times = [f"{r['timestamp']:.1f}s" for r in frame_results if r["score"] > 0.60]

    if not loaded_custom:
        verdict = "Inconclusive"
        confidence = 0.5
        reasons = [
            "Custom fine-tuned weights not loaded; running fallback baseline check",
            f"Sampled {len(frame_results)} frames ({faces_found} with detected faces)",
        ]
    elif agg_score > 0.60:
        verdict = "Potentially Manipulated"
        confidence = agg_score
        reasons = [
            f"High manipulation cues detected across sampled frames (top-frame score {agg_score * 100:.1f}%)",
        ]
        if flagged_times:
            reasons.append(f"Suspicious timestamps flagged at: {', '.join(flagged_times[:4])}")
    elif agg_score < 0.40:
        verdict = "Likely Authentic"
        confidence = 1.0 - agg_score
        reasons = [
            f"Consistent visual features across {len(frame_results)} sampled frames ({(1.0 - agg_score) * 100:.1f}% authenticity score)",
        ]
    else:
        verdict = "Inconclusive"
        confidence = round(1.0 - abs(agg_score - 0.5) * 2, 4)
        reasons = [
            f"Frame-level manipulation score ({agg_score * 100:.1f}%) lies in the uncertain range",
        ]

    if loaded_custom:
        if faces_found == 0:
            reasons.append("No clear frontal faces detected; full-frame visual artifacts evaluated")
        else:
            reasons.append(f"Facial region analysis completed on {faces_found}/{len(frame_results)} sampled frames")

        if score_std > 0.18:
            reasons.append(f"High inter-frame score variance (std={score_std:.2f}) indicates temporal inconsistency")

    return {
        "media_type": "video",
        "verdict": verdict,
        "confidence": round(float(confidence), 4),
        "manipulation_score": round(float(agg_score), 4),
        "reasons": reasons,
        "evidence_path": evidence_out_path,
    }
