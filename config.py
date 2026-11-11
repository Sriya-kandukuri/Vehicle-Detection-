from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "change-this-key-before-production")
    DATABASE = BASE_DIR / "instance" / "vehiclevision.db"
    UPLOAD_FOLDER = BASE_DIR / "uploads" / "input"
    OUTPUT_FOLDER = BASE_DIR / "uploads" / "output"
    DATASET_FOLDER = BASE_DIR / "uploads" / "datasets"
    # Flask/Werkzeug streams large file bodies to temporary storage instead of
    # holding the complete upload in RAM. The default supports common Kaggle
    # vehicle datasets and can be changed without editing the source code.
    MAX_UPLOAD_MB = max(1, int(os.getenv("MAX_UPLOAD_MB", "2048")))
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
    ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
    ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
