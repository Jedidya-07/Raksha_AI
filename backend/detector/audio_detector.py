import os
import joblib
import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_audio_model(model_path: str | None = None):
    """Loads a saved scikit-learn audio classifier if present; otherwise returns None."""
    if model_path and os.path.exists(model_path):
        clf = joblib.load(model_path)
        return clf, True
    return None, False


def _extract_audio_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Extracts mean/std of MFCCs, spectral centroid, and spectral flatness."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)

    feats = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        [float(np.mean(centroid)), float(np.std(centroid))],
        [float(np.mean(flatness)), float(np.std(flatness))],
    ])
    return feats.reshape(1, -1)


def _save_spectrogram_evidence(y: np.ndarray, sr: int, out_path: str) -> None:
    """Saves a clean Mel-spectrogram image to static/evidence_*.png."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(8, 3.5), dpi=120)
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Audio Mel-Spectrogram Forensic View")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def analyze_audio(
    filepath: str,
    clf,
    loaded_custom: bool,
    evidence_out_path: str,
    max_duration_sec: float = 15.0,
) -> dict:
    """Analyzes audio for synthetic/spoof cues and generates a Mel-spectrogram image."""
    y, sr = librosa.load(filepath, sr=22050, mono=True, duration=max_duration_sec)

    if y is None or len(y) == 0:
        raise ValueError("Audio file appears empty or unreadable.")

    rms_energy = float(np.mean(librosa.feature.rms(y=y)))
    _save_spectrogram_evidence(y, sr, evidence_out_path)

    feats = _extract_audio_features(y, sr)
    mean_flatness = float(feats[0, -2])

    if rms_energy < 1e-4:
        return {
            "media_type": "audio",
            "verdict": "Inconclusive",
            "confidence": 0.25,
            "manipulation_score": 0.5,
            "reasons": [
                "Audio signal is near-silent; insufficient speech/signal energy for reliable check",
            ],
            "evidence_path": evidence_out_path,
        }

    if loaded_custom and clf is not None:
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(feats)[0]
            classes = list(getattr(clf, "classes_", [0, 1]))
            fake_idx = classes.index(1) if 1 in classes else (len(classes) - 1)
            fake_prob = float(probs[fake_idx])
        else:
            pred = int(clf.predict(feats)[0])
            fake_prob = 0.85 if pred == 1 else 0.15

        if fake_prob > 0.60:
            verdict = "Potentially Manipulated"
            confidence = fake_prob
            reasons = [
                f"Audio classifier flagged synthetic speech/spoof patterns ({fake_prob * 100:.1f}% manipulation score)",
            ]
        elif fake_prob < 0.40:
            verdict = "Likely Authentic"
            confidence = 1.0 - fake_prob
            reasons = [
                f"Spectral and MFCC features align with natural audio ({(1.0 - fake_prob) * 100:.1f}% authenticity confidence)",
            ]
        else:
            verdict = "Inconclusive"
            confidence = round(1.0 - abs(fake_prob - 0.5) * 2, 4)
            reasons = [
                f"Audio manipulation score ({fake_prob * 100:.1f}%) falls in the uncertain range",
            ]
    else:
        fake_prob = 0.5
        verdict = "Inconclusive"
        confidence = 0.5
        reasons = [
            "Trained audio model (audio_model.pkl) not loaded; Mel-spectrogram generated for inspection",
        ]

    reasons.append(f"Mean spectral flatness: {mean_flatness:.4f} | RMS energy: {rms_energy:.4f}")

    return {
        "media_type": "audio",
        "verdict": verdict,
        "confidence": round(float(confidence), 4),
        "manipulation_score": round(float(fake_prob), 4),
        "reasons": reasons,
        "evidence_path": evidence_out_path,
    }
