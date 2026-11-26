import io
import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import torch
import torch.nn as nn
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_image_model(weights_path: str | None = None):
    """Loads fine-tuned MobileNetV3 weights if present; falls back cleanly to ImageNet weights."""
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    fake_idx = 0
    loaded_custom = False

    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            class_to_idx = checkpoint.get("class_to_idx", {"FAKE": 0, "REAL": 1})
            fake_idx = int(class_to_idx.get("FAKE", 0))
        else:
            model.load_state_dict(checkpoint)
        loaded_custom = True
    else:
        pretrained = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        pretrained.classifier[3] = nn.Linear(pretrained.classifier[3].in_features, 2)
        model.load_state_dict(pretrained.state_dict())

    model.to(DEVICE).eval()
    return model, fake_idx, loaded_custom


def generate_ela_evidence(pil_img: Image.Image, out_path: str) -> float:
    """Generates a side-by-side Original vs. Error Level Analysis (ELA) image and returns mean diff."""
    rgb = pil_img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")

    diff = ImageChops.difference(rgb, resaved)
    mean_residual = float(np.asarray(diff, dtype=np.float32).mean())

    max_val = max(ex[1] for ex in diff.getextrema())
    scale = 255.0 / max_val if max_val > 0 else 1.0
    ela_vis = ImageEnhance.Brightness(diff).enhance(scale)

    tile_w, tile_h = 320, 320
    left = rgb.resize((tile_w, tile_h))
    right = ela_vis.resize((tile_w, tile_h))

    canvas = Image.new("RGB", (tile_w * 2, tile_h))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (tile_w, 0))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)
    return mean_residual


def analyze_image(
    filepath: str,
    model,
    fake_idx: int,
    evidence_out_path: str,
    loaded_custom: bool = True,
) -> dict:
    """Runs image manipulation scoring + ELA visual evidence generation."""
    pil_img = Image.open(filepath).convert("RGB")
    width, height = pil_img.size

    tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        fake_prob = float(probs[fake_idx].item())

    ela_residual = generate_ela_evidence(pil_img, evidence_out_path)

    # If custom weights aren't loaded yet, keep verdict honest as Inconclusive for demo safety
    if not loaded_custom:
        verdict = "Inconclusive"
        confidence = 0.5
        reasons = [
            "Custom fine-tuned weights not yet loaded; running fallback baseline check",
            f"Mean compression residual (ELA): {ela_residual:.2f}",
        ]
    elif fake_prob > 0.60:
        verdict = "Potentially Manipulated"
        confidence = fake_prob
        reasons = [
            f"Classifier detected synthetic/manipulation cues ({fake_prob * 100:.1f}% manipulation probability)",
        ]
    elif fake_prob < 0.40:
        verdict = "Likely Authentic"
        confidence = 1.0 - fake_prob
        reasons = [
            f"Classifier found patterns consistent with authentic imagery ({(1.0 - fake_prob) * 100:.1f}% authenticity confidence)",
        ]
    else:
        verdict = "Inconclusive"
        confidence = round(1.0 - abs(fake_prob - 0.5) * 2, 4)
        reasons = [
            f"Manipulation score ({fake_prob * 100:.1f}%) falls in the uncertain band",
        ]

    if loaded_custom:
        if ela_residual > 8.5:
            reasons.append(f"Elevated Error Level Analysis residual ({ela_residual:.2f}) suggests re-saving or local edits")
        else:
            reasons.append(f"Consistent Error Level Analysis residual ({ela_residual:.2f}) across regions")

    if width < 128 or height < 128:
        reasons.append(f"Low resolution ({width}x{height}px) may reduce detection accuracy")

    return {
        "media_type": "image",
        "verdict": verdict,
        "confidence": round(float(confidence), 4),
        "manipulation_score": round(float(fake_prob), 4),
        "reasons": reasons,
        "evidence_path": evidence_out_path,
    }
