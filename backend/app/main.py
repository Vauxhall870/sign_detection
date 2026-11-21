import asyncio
import json
import os
import time

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import library
from config import (
    CONFIDENCE_THRESHOLD,
    FEWSHOT_MAX_CLIPS,
    FEWSHOT_MIN_CLIPS,
    FEWSHOT_THRESHOLD,
    ISLR_MODEL_PATH,
    SIGN_DIR,
)
from ml import fewshot
from ml.landmarks import (
    decode_data_url,
    detect,
    extract_keypoints,
    extract_raw543,
    has_hands,
    make_holistic,
)
from ml.predict import FrameBuffer, recognizer

app = FastAPI(title="Sign Language Translator")

# Vite dev server (5173) and `vite preview` of the production build (4173);
# set CORS_ORIGINS (comma-separated) when serving the frontend from elsewhere
CORS_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=SIGN_DIR), name="media")


class TextRequest(BaseModel):
    text: str


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_trained": recognizer.ready,
        "signs": recognizer.labels,
        "model": recognizer.kind,
        "sequence_length": recognizer.sequence_length,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


@app.get("/api/vocabulary")
def vocabulary():
    """What the system can recognise, and what it can display."""
    return {
        "recognizable": recognizer.labels,
        "displayable": library.available_words(),
    }


@app.post("/api/text-to-sign")
def text_to_sign(req: TextRequest):
    return {"text": req.text, "sequence": library.translate(req.text)}


# ---------------------------------------------------------------------------
# Few-shot "teach a new sign" (see ml/fewshot.py for how it works)
# ---------------------------------------------------------------------------
MIN_FRAMES_PER_CLIP = 8      # fewer than this is too short to be a sign
MAX_FRAMES_PER_CLIP = 150    # guards against huge uploads
MIN_HAND_FRACTION = 0.3      # a clip where hands are rarely seen can't teach a hand sign


class TeachRequest(BaseModel):
    name: str
    # each clip is a list of base64 JPEG frames (data URLs), in time order
    clips: list[list[str]]


def clip_to_window(frames: list[str], clip_no: int) -> np.ndarray:
    """One recorded clip (JPEG frames) -> (45, 543, 3) landmark window.
    Runs MediaPipe on each frame, exactly like the live recogniser does."""
    holistic = make_holistic()   # fresh instance: Holistic tracks across frames, so don't share between clips
    try:
        track, with_hands = [], 0
        for data_url in frames[:MAX_FRAMES_PER_CLIP]:
            frame = decode_data_url(data_url)
            if frame is None:
                continue
            results = detect(frame, holistic)
            track.append(extract_raw543(results))
            with_hands += has_hands(results)
    finally:
        holistic.close()
    if len(track) < MIN_FRAMES_PER_CLIP:
        raise HTTPException(422, f"Clip {clip_no}: only {len(track)} readable frames (need {MIN_FRAMES_PER_CLIP}).")
    if with_hands / len(track) < MIN_HAND_FRACTION:
        raise HTTPException(422, f"Clip {clip_no}: hands were visible in too few frames. Keep your hands in view.")
    return fewshot.track_to_window(np.stack(track))


@app.post("/api/teach")
def teach(req: TeachRequest):
    """Teach a new sign from 3-5 recorded clips. Stores one averaged prototype."""
    if not ISLR_MODEL_PATH.exists():
        raise HTTPException(503, "Pretrained model missing; few-shot teaching needs it.")
    if not FEWSHOT_MIN_CLIPS <= len(req.clips) <= FEWSHOT_MAX_CLIPS:
        raise HTTPException(422, f"Need {FEWSHOT_MIN_CLIPS}-{FEWSHOT_MAX_CLIPS} clips, got {len(req.clips)}.")
    windows = [clip_to_window(frames, i + 1) for i, frames in enumerate(req.clips)]
    try:
        return fewshot.add_sign(req.name, windows)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/prototypes")
def prototypes():
    """Signs taught so far, and the similarity needed to match them."""
    return {"signs": fewshot.list_signs(), "threshold": FEWSHOT_THRESHOLD}


@app.delete("/api/prototypes/{name}")
def delete_prototype(name: str):
    try:
        removed = fewshot.delete_sign(name)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if not removed:
        raise HTTPException(404, f"No taught sign called '{name}'.")
    return {"deleted": name}


@app.websocket("/ws/recognize")
async def recognize(ws: WebSocket):
    """Client streams base64 JPEG frames; we stream back predictions.

    MediaPipe Holistic is stateful per video stream, so each connection gets
    its own instance and its own rolling window.
    """
    await ws.accept()
    buffer = FrameBuffer(recognizer.window_seconds, recognizer.sequence_length)
    holistic = make_holistic()
    extract = extract_raw543 if recognizer.kind == "islr" else extract_keypoints

    def process(data_url: str, t: float) -> dict | None:
        """CPU-bound: runs in a worker thread so the event loop stays free."""
        frame = decode_data_url(data_url)
        if frame is None:
            return None
        results = detect(frame, holistic)
        window = buffer.push(extract(results), t)
        hands = has_hands(results)
        if window is None:
            return {"type": "buffering", "filled": buffer.filled, "needed": buffer.sequence_length, "hands": hands}
        label, confidence, scores = recognizer.predict(window)
        return {
            "type": "prediction",
            "label": label,
            "confidence": confidence,
            "scores": scores,
            "hands": hands,
            "accepted": buffer.accept(label, confidence),
        }

    # Holistic runs slower than the client sends, so frames are not queued:
    # the receiver keeps only the newest one and the processor picks it up
    # when free. Queuing every frame makes latency grow without bound.
    state = {"frame": None, "reset": False, "closed": False}
    wake = asyncio.Event()

    async def receive():
        try:
            while True:
                message = await ws.receive_text()
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    payload = None
                if not isinstance(payload, dict):
                    payload = {"frame": message}
                if payload.get("action") == "reset":
                    state["reset"], state["frame"] = True, None
                elif isinstance(payload.get("frame"), str) and payload["frame"]:
                    state["frame"] = (payload["frame"], time.monotonic())
                wake.set()
        except WebSocketDisconnect:
            pass
        finally:
            state["closed"] = True
            wake.set()

    receiver = None
    try:
        if not recognizer.ready:
            await ws.send_json({"type": "error", "message": "No trained model yet. Run ml/train.py."})
            await ws.close()
            return

        await asyncio.to_thread(recognizer.load)
        await ws.send_json({"type": "ready", "signs": recognizer.labels})

        receiver = asyncio.create_task(receive())
        while True:
            await wake.wait()
            wake.clear()
            if state["closed"]:
                break
            if state["reset"]:
                state["reset"] = False
                buffer.reset()
                await ws.send_json({"type": "reset"})
            item, state["frame"] = state["frame"], None
            if item is None:
                continue
            msg = await asyncio.to_thread(process, *item)
            # the client may have hung up while Holistic was running;
            # sending after close raises in Starlette
            if state["closed"]:
                break
            if msg is not None:
                await ws.send_json(msg)

    except WebSocketDisconnect:
        pass
    finally:
        if receiver is not None:
            receiver.cancel()
        holistic.close()
