"""Lyrics embedding (catalogue section 16): stored lyrics, else the Whisper transcript from `stems`.

Tracks with neither get a zero vector with has_text = 0, so dimensions stay fixed.
"""

import numpy as np

from .. import db

EXTRACTOR = "lyrics"
VERSION = 1
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Stage:
    name = "lyrics"
    outputs = [(EXTRACTOR, VERSION)]
    needs_audio = False

    def load(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL, device="cpu")

    def ready(self, con, track) -> bool:
        """Wait for the stems stage when the transcript is the only possible text."""
        return bool(track["lyrics"]) or db.get_transcript(con, track["id"]) is not None

    def process_text(self, con, track) -> dict:
        transcript = db.get_transcript(con, track["id"]) or ""
        text = track["lyrics"] or transcript
        emb = self.model.encode(text) if text.strip() else np.zeros(self.model.get_sentence_embedding_dimension())
        names = [f"lyrics_{i}" for i in range(len(emb))] + ["has_text", "from_transcript"]
        vec = np.r_[emb, float(bool(text.strip())), float(not track["lyrics"] and bool(transcript.strip()))]
        return {EXTRACTOR: (names, vec.astype(np.float32))}
