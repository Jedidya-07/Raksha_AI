import io
import json
import os
import shutil
import uuid
from datetime import datetime
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from detector.audio_detector import analyze_audio, load_audio_model
from detector.image_detector import analyze_image, load_image_model
from detector.video_detector import analyze_video

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "veritrust.db")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Database Setup (SQLite + SQLAlchemy) ---
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class VerificationRecord(Base):
    __tablename__ = "verifications"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    media_type = Column(String, nullable=False)
    verdict = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    manipulation_score = Column(Float, nullable=False)
    reasons_json = Column(Text, nullable=False)
    evidence_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# --- FastAPI App ---
app = FastAPI(title="VERITRUST AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Load Detectors Once on Startup ---
IMAGE_MODEL, FAKE_IDX, LOADED_CUSTOM_IMAGE = load_image_model(
    os.path.join(MODELS_DIR, "image_model.pt")
)
AUDIO_MODEL, LOADED_CUSTOM_AUDIO = load_audio_model(
    os.path.join(MODELS_DIR, "audio_model.pkl")
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".m4a", ".flac"}


class VerificationOut(BaseModel):
    id: int
    filename: str
    media_type: str
    verdict: str
    confidence: float
    manipulation_score: float
    reasons: List[str]
    evidence_image_url: str
    created_at: str


def _detect_media_type(ext: str) -> str:
    ext = ext.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext}")


def _to_schema(rec: VerificationRecord) -> VerificationOut:
    evidence_name = os.path.basename(rec.evidence_path)
    return VerificationOut(
        id=rec.id,
        filename=rec.filename,
        media_type=rec.media_type,
        verdict=rec.verdict,
        confidence=rec.confidence,
        manipulation_score=rec.manipulation_score,
        reasons=json.loads(rec.reasons_json),
        evidence_image_url=f"/static/{evidence_name}",
        created_at=rec.created_at.isoformat(),
    )


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "custom_image_model_loaded": LOADED_CUSTOM_IMAGE,
        "custom_audio_model_loaded": LOADED_CUSTOM_AUDIO,
    }


@app.post("/api/analyze", response_model=VerificationOut)
async def analyze_media(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = os.path.splitext(file.filename)[1]
    media_type = _detect_media_type(ext)

    unique_id = uuid.uuid4().hex[:10]
    safe_upload_name = f"{unique_id}{ext.lower()}"
    upload_path = os.path.join(UPLOAD_DIR, safe_upload_name)

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    evidence_filename = f"evidence_{unique_id}.png"
    evidence_out_path = os.path.join(STATIC_DIR, evidence_filename)

    try:
        if media_type == "image":
            result = analyze_image(
                upload_path, IMAGE_MODEL, FAKE_IDX, evidence_out_path, LOADED_CUSTOM_IMAGE
            )
        elif media_type == "video":
            result = analyze_video(
                upload_path, IMAGE_MODEL, FAKE_IDX, evidence_out_path, LOADED_CUSTOM_IMAGE
            )
        else:
            result = analyze_audio(
                upload_path, AUDIO_MODEL, LOADED_CUSTOM_AUDIO, evidence_out_path
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(exc)}")

    db = SessionLocal()
    try:
        record = VerificationRecord(
            filename=file.filename,
            media_type=result["media_type"],
            verdict=result["verdict"],
            confidence=result["confidence"],
            manipulation_score=result["manipulation_score"],
            reasons_json=json.dumps(result["reasons"]),
            evidence_path=result["evidence_path"],
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return _to_schema(record)
    finally:
        db.close()


@app.get("/api/history", response_model=List[VerificationOut])
def get_history(limit: int = 25):
    db = SessionLocal()
    try:
        rows = (
            db.query(VerificationRecord)
            .order_by(VerificationRecord.id.desc())
            .limit(limit)
            .all()
        )
        return [_to_schema(r) for r in rows]
    finally:
        db.close()


@app.get("/api/report/{record_id}")
def download_pdf_report(record_id: int):
    db = SessionLocal()
    try:
        rec = db.query(VerificationRecord).filter(VerificationRecord.id == record_id).first()
        if not rec:
            raise HTTPException(status_code=404, detail="Verification record not found.")

        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        width, height = letter

        y = height - 60
        c.setFont("Helvetica-Bold", 18)
        c.drawString(50, y, "VERITRUST AI - Digital Media Verification Report")

        y -= 28
        c.setFont("Helvetica", 11)
        c.drawString(50, y, f"Report ID: #{rec.id}   |   Analyzed (UTC): {rec.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        y -= 18
        c.drawString(50, y, f"File Name: {rec.filename}   |   Modality: {rec.media_type.upper()}")
        y -= 18
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"Verdict: {rec.verdict}   |   Confidence: {rec.confidence * 100:.1f}%")

        y -= 28
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Forensic Findings:")
        c.setFont("Helvetica", 10)
        for reason in json.loads(rec.reasons_json):
            y -= 16
            c.drawString(62, y, f"- {reason[:95]}")

        if os.path.exists(rec.evidence_path):
            y -= 210
            if y < 60:
                c.showPage()
                y = height - 240
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y + 185, "Visual Evidence:")
            c.drawImage(rec.evidence_path, 50, y, width=420, height=175, preserveAspectRatio=True)

        c.save()
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="veritrust_report_{rec.id}.pdf"'},
        )
    finally:
        db.close()
