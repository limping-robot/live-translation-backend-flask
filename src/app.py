import json
import os
import datetime
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import jwt

from flask import Flask, jsonify, request
from flask_sock import Sock
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ---------------- Config ----------------
CUDA_AVAILABLE = torch.cuda.is_available()

TRANSCRIPTION_MODEL_NAME = os.getenv("TRANSCRIPTION_MODEL_NAME", "base")
TRANSCRIPTION_DEVICE = os.getenv("TRANSCRIPTION_DEVICE", "cuda" if CUDA_AVAILABLE else "cpu")
TRANSCRIPTION_DATATYPE = "float32" if CUDA_AVAILABLE else "int8"

TRANSLATION_MODEL_NAME = os.getenv("TRANSLATION_MODEL", "Helsinki-NLP/opus-mt-tl-en")
TRANSLATION_DEVICE = os.getenv("ASR_DEVICE", "cuda" if CUDA_AVAILABLE else "cpu")

TARGET_SR = 16000

# Static credentials (change these or set via env)
STATIC_USERNAME = os.getenv("APP_USERNAME", "percy")
STATIC_PASSWORD = os.getenv("APP_PASSWORD", "p3nGu1n")

# JWT config
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "sG0q8X2P9QeM4ZpO2u1vV4iC8wLh6RrQ_JsE7uQxT0A")
JWT_ALGORITHM = "HS256"
JWT_EXPIRES_HOURS = int(os.getenv("JWT_EXPIRES_HOURS", "12"))
# --------------------------------------


# Module-level model storage (loaded once in create_app)
whisper_model: Optional[WhisperModel] = None
translation_tokenizer: Optional[AutoTokenizer] = None
translation_model: Optional[AutoModelForSeq2SeqLM] = None


def create_access_token(identity: str) -> str:
    """Create a signed JWT for the given identity."""
    now = datetime.datetime.utcnow()
    payload = {
        "sub": identity,
        "iat": now,
        "exp": now + datetime.timedelta(hours=JWT_EXPIRES_HOURS),
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    # PyJWT>=2 returns a str already; if bytes, decode:
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token


def verify_token(token: str) -> Optional[str]:
    """Return username from token if valid, else None."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def pcm16_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Simple linear resampler. Good enough for speech prototyping."""
    if src_sr == dst_sr or x.size == 0:
        return x
    ratio = dst_sr / float(src_sr)
    n = int(round(x.size * ratio))
    if n <= 1:
        return x[:1]
    src_idx = np.linspace(0, x.size - 1, num=n, dtype=np.float32)
    lo = np.floor(src_idx).astype(np.int32)
    hi = np.minimum(lo + 1, x.size - 1)
    w = src_idx - lo
    return (1.0 - w) * x[lo] + w * x[hi]


@torch.no_grad()
def translate_tl_to_en(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    inputs = translation_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(TRANSLATION_DEVICE)
    out = translation_model.generate(**inputs, max_new_tokens=256, num_beams=4)
    return translation_tokenizer.decode(out[0], skip_special_tokens=True).strip()


@dataclass
class UtteranceBuffer:
    utt_id: str
    sample_rate: int
    buf: bytearray


def create_app(config=None):
    """
    Application factory function.
    
    Args:
        config: Optional configuration object (for future use)
    
    Returns:
        Flask application instance
    """
    global whisper_model, translation_tokenizer, translation_model
    
    app = Flask(__name__)
    sock = Sock(app)
    
    # Load models once (only on first factory call)
    if whisper_model is None:
        print(
            f"Loading models: TRANSCRIPTION_MODEL_NAME={TRANSCRIPTION_MODEL_NAME}, "
            f"TRANSCRIPTION_DEVICE={TRANSCRIPTION_DEVICE}, "
            f"TRANSCRIPTION_DATATYPE={TRANSCRIPTION_DATATYPE}, "
            f"TRANSLATION_MODEL_NAME={TRANSLATION_MODEL_NAME}, "
            f"TRANSLATION_DEVICE={TRANSLATION_DEVICE}"
        )
        
        whisper_model = WhisperModel(
            TRANSCRIPTION_MODEL_NAME,
            device=TRANSCRIPTION_DEVICE,
            compute_type=TRANSCRIPTION_DATATYPE,
        )
        translation_tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
        translation_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_NAME).to(TRANSLATION_DEVICE)
        translation_model.eval()
    
    @app.post("/login")
    def login():  # pyright: ignore[reportUnusedFunction]
        """Simple static-credential login: returns JWT if username/password match."""
        data = request.get_json(silent=True) or {}
        username = data.get("username")
        password = data.get("password")

        print(f"Authenticating username={username}")

        if username == STATIC_USERNAME and password == STATIC_PASSWORD:
            token = create_access_token(username)
            print(f"Successfully authenticated username={username}")
            return jsonify({"access_token": token})

        print(f"Failed authenticating username={username} password={password}")
        return jsonify({"error": "Invalid credentials"}), 401

    @app.get("/health")
    def health():  # pyright: ignore[reportUnusedFunction]
        # Keep it cheap: confirm process is up and models are loaded.
        return jsonify({
            "ok": True,
            "TRANSCRIPTION_MODEL_NAME": TRANSCRIPTION_MODEL_NAME,
            "TRANSCRIPTION_DEVICE": TRANSCRIPTION_DEVICE,
            "TRANSCRIPTION_DATATYPE": TRANSCRIPTION_DATATYPE,
            "TRANSLATION_MODEL_NAME": TRANSLATION_MODEL_NAME,
            "TRANSLATION_DEVICE": TRANSLATION_DEVICE,
        })

    @sock.route("/ws")
    def ws_handler(ws):  # pyright: ignore[reportUnusedFunction]
        """
        WebSocket handler with JWT auth.

        The client must connect as: ws://host/ws?token=JWT_HERE
        """
        # Get token from query string
        token = request.args.get("token")
        user = verify_token(token) if token else None

        if not user:
            # Minimal feedback then close
            try:
                ws.send(json.dumps({
                    "type": "error",
                    "message": "unauthorized",
                }))
            except Exception:
                pass
            ws.close()
            return

        current: Optional[UtteranceBuffer] = None

        while True:
            msg = ws.receive()
            if msg is None:
                break

            # JSON control messages
            if isinstance(msg, str):
                obj = json.loads(msg)
                t = obj.get("type")

                if t == "utt_start":
                    current = UtteranceBuffer(
                        utt_id=obj["uttId"],
                        sample_rate=int(obj["sampleRate"]),
                        buf=bytearray(),
                    )

                elif t == "utt_end":
                    if current is None or obj.get("uttId") != current.utt_id:
                        continue

                    pcm_bytes = bytes(current.buf)
                    utt_id = current.utt_id
                    src_sr = current.sample_rate
                    current = None

                    audio = pcm16_bytes_to_float32(pcm_bytes)

                    # Resample to 16k for consistency
                    if src_sr != TARGET_SR:
                        audio = resample_linear(audio, src_sr, TARGET_SR)

                    segments, info = whisper_model.transcribe(
                        audio,
                        language="tl",   # expected Tagalog speech input
                        beam_size=5,
                    )
                    tl = "".join(s.text for s in segments).strip()
                    en = translate_tl_to_en(tl) if tl else ""

                    ws.send(json.dumps({
                        "type": "result",
                        "uttId": utt_id,
                        "en": en,
                        "tl": tl,
                    }))

                elif t == "ping":
                    ws.send(json.dumps({"type": "pong"}))

            # Binary PCM chunk
            else:
                if current is not None:
                    current.buf.extend(msg)
    
    return app


# Create app instance for gunicorn/WSGI servers
app = create_app()


if __name__ == "__main__":
    # Dev server is OK for local testing.
    # For production websockets, run behind a proper server.
    app.run(host="0.0.0.0", port=5000, threaded=True)
