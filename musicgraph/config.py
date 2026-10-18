import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.environ.get("MG_DATA_DIR", ROOT / "data"))
DB_PATH = DATA_DIR / "music.db"
AUDIO_DIR = DATA_DIR / "audio"
SPEC_DIR = DATA_DIR / "spectrograms"
MAPS_DIR = DATA_DIR / "maps"

# Audio analysis parameters shared by every stage
SR = 22050
N_FFT = 2048
HOP = 512
N_MELS = 128

JAMENDO_CLIENT_ID = os.environ.get("JAMENDO_CLIENT_ID", "")


def ensure_dirs() -> None:
    for d in (DATA_DIR, AUDIO_DIR, SPEC_DIR, MAPS_DIR):
        d.mkdir(parents=True, exist_ok=True)
