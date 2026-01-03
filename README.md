## Overview
A Flask backend for real-time speech transcription and translation. It transcribes English speech and translates it to Tagalog (Filipino).

## Features

1. Authentication
   - `/login` endpoint with static credentials (username/password)
   - JWT tokens for WebSocket connections
   - Tokens expire after 12 hours (configurable)

2. Real-time audio processing via WebSocket (`/ws`)
   - Accepts PCM16 audio chunks
   - Handles utterance start/end messages
   - Resamples audio to 16kHz if needed

3. Speech transcription
   - Uses `faster-whisper` (Whisper model)
   - Expects English input
   - Model configurable via env vars (defaults to "base")

4. Translation
   - Translates English → Tagalog using Helsinki-NLP's `opus-mt-en-tl`
   - Runs on GPU if available, otherwise CPU

5. Health check
   - `/health` endpoint shows model configuration and device info

## Technical details
- Uses CUDA when available (falls back to CPU)
- Models loaded once at startup
- WebSocket protocol with JSON control messages and binary audio data
- Dockerized with gunicorn for production

## Workflow
1. Client logs in → receives JWT
2. Client connects to `/ws?token=JWT`
3. Client sends `utt_start` → streams PCM16 audio → sends `utt_end`
4. Server transcribes → translates → sends back English text + Tagalog translation
