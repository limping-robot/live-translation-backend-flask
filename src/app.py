import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from flask import Flask
from flask import jsonify
from flask_sock import Sock
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ---------------- Config ----------------
CUDA_AVAILABLE = torch.cuda.is_available()

TRANSCRIPTION_MODEL_NAME = os.getenv("TRANSCRIPTION", "base")
TRANSCRIPTION_DEVICE = os.getenv("TRANSCRIPTION_DEVICE", "cuda" if CUDA_AVAILABLE else "cpu")
TRANSCRIPTION_DATATYPE = "float32" if CUDA_AVAILABLE else "int8"

TRANSLATION_MODEL_NAME = os.getenv("TRANSLATION_MODEL", "Helsinki-NLP/opus-mt-en-tl")
TRANSLATION_DEVICE = os.getenv("ASR_DEVICE", "cuda" if CUDA_AVAILABLE else "cpu")

TARGET_SR = 16000
# --------------------------------------


app = Flask(__name__)
sock = Sock(app)

@app.get("/health")
def health():
    # Keep it cheap: confirm process is up and models are loaded.
    return jsonify({
        "ok": True,
        "TRANSCRIPTION_MODEL_NAME": TRANSCRIPTION_MODEL_NAME,
        "TRANSCRIPTION_DEVICE": TRANSCRIPTION_DEVICE,
        "TRANSCRIPTION_DATATYPE": TRANSCRIPTION_DATATYPE,
        "TRANSLATION_MODEL_NAME": TRANSLATION_MODEL_NAME,
        "TRANSLATION_DEVICE": TRANSLATION_DEVICE,
    })

print(f"TRANSCRIPTION_MODEL_NAME={TRANSCRIPTION_MODEL_NAME}, " +
        f"TRANSCRIPTION_DEVICE={TRANSCRIPTION_DEVICE}, " +
        f"TRANSCRIPTION_DATATYPE={TRANSCRIPTION_DATATYPE}, " +
        f"TRANSLATION_MODEL_NAME={TRANSLATION_MODEL_NAME}, " +
        f"TRANSLATION_DEVICE={TRANSLATION_DEVICE}")

# Load models once (global)
whisper_model = \
    WhisperModel(TRANSCRIPTION_MODEL_NAME, device=TRANSCRIPTION_DEVICE, compute_type=TRANSCRIPTION_DATATYPE)
translation_tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_NAME).to(TRANSLATION_DEVICE)
translation_model.eval()

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
def translate_en_to_tl(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    inputs = translation_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(TRANSLATION_DEVICE)
    out = translation_model.generate(**inputs, max_new_tokens=256, num_beams=4)
    return translation_tokenizer.decode(out[0], skip_special_tokens=True).strip()


@dataclass
class UtteranceBuffer:
    utt_id: str
    sample_rate: int
    buf: bytearray


@sock.route("/ws")
def ws_handler(ws):
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
                    language="en",   # expected English speech input
                    beam_size=5,
                )
                en = "".join(s.text for s in segments).strip()
                tl = translate_en_to_tl(en) if en else ""

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


if __name__ == "__main__":
    # Dev server is OK for local testing.
    # For production websockets, run behind a proper server.
    app.run(host="0.0.0.0", port=5000, threaded=True)
