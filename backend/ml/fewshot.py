"""Few-shot "teach a new sign" on top of the pretrained ISLR model - no retraining.

How it works (the viva explanation):
  1. The pretrained model turns a sign (a 45-frame landmark window) into 3 x 384
     numbers just before its final 250-way classifier. That vector, the
     "embedding", describes WHAT was signed, so the same sign gives similar
     vectors and different signs give different ones (see ml/check_embeddings.py).
  2. To teach a new sign we record 3-5 clips, embed each, and AVERAGE them into
     one vector: the sign's "prototype". Saving that is all "training" is.
  3. To recognise, we embed the live window and compare it with every taught
     prototype by cosine similarity. The best one wins if it is similar enough.

The pretrained model itself is never changed; this file only READS from it.
"""
import json
import os
import re
import threading
import time

import numpy as np

from config import (
    FEWSHOT_EVAL_DIR,
    FEWSHOT_EVAL_MAX_CLIPS,
    FEWSHOT_MAX_CLIPS,
    FEWSHOT_MIN_CLIPS,
    FEWSHOT_PROTOTYPES_PATH,
    FEWSHOT_THRESHOLD,
    FEWSHOT_WINDOW_FRAMES,
    ISLR_MODEL_PATH,
)

PROTOTYPES_PATH = FEWSHOT_PROTOTYPES_PATH   # module-level so tests can point it elsewhere


def unit(v: np.ndarray) -> np.ndarray:
    """Scale to length 1, so a plain dot product of two vectors IS their cosine similarity."""
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def track_to_window(track: np.ndarray) -> np.ndarray:
    """(any number of frames, 543, 3) -> exactly (45, 543, 3), picking evenly spaced frames.
    A recorded clip is one whole sign, so it is stretched over the model's fixed window."""
    idx = np.linspace(0, len(track) - 1, FEWSHOT_WINDOW_FRAMES).round().astype(int)
    return track[idx].astype(np.float32)


# --------------------------------------------------------------------------
# embedding: window -> vector
# --------------------------------------------------------------------------
class Embedder:
    """Runs the ISLR TFLite model on its OWN interpreter (the live recogniser's is
    untouched) and reads the tensors just before each classifier layer."""

    def __init__(self):
        import tensorflow as tf  # imported lazily: TF startup is slow

        # experimental_preserve_all_tensors keeps intermediate results readable
        # after invoke(); by default TFLite reuses that memory and they are lost.
        self._it = tf.lite.Interpreter(str(ISLR_MODEL_PATH), experimental_preserve_all_tensors=True)
        self._run = self._it.get_signature_runner("serving_default")
        # The model is an ensemble of 3 sub-networks. Each ends with
        # global_average_pooling1d (average over time) -> 384 numbers -> classifier
        # (a fully-connected layer with a 250 x 384 weight matrix). So "the
        # embedding" = the input of each such classifier layer. Found by shape
        # rather than hard-coding tensor indices.
        shapes = {d["index"]: list(d["shape"]) for d in self._it.get_tensor_details()}
        self._pool_ids = [
            op["inputs"][0] for op in self._it._get_ops_details()
            if op["op_name"] == "FULLY_CONNECTED"
            and any(i >= 0 and shapes[i] == [250, 384] for i in op["inputs"])
        ]
        if len(self._pool_ids) != 3:
            raise RuntimeError(f"expected 3 pooled tensors, found {len(self._pool_ids)}")
        self._lock = threading.Lock()   # one TFLite interpreter is not re-entrant

    def parts(self, window: np.ndarray) -> np.ndarray:
        """-> (3, 384): one raw embedding per sub-network."""
        with self._lock:
            self._run(inputs=window.astype(np.float32))
            return np.stack([self._it.get_tensor(i)[0] for i in self._pool_ids])


_embedder: Embedder | None = None
_embedder_lock = threading.Lock()


def _get_embedder() -> Embedder:
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            _embedder = Embedder()
        return _embedder


def get_embedding(window: np.ndarray) -> np.ndarray:
    """(45, 543, 3) raw landmark window -> (384,) L2-normalised embedding.

    The model has 3 sub-networks, each giving 384 numbers; we average them
    (check_embeddings.py showed this is as good as concatenating) and normalise."""
    if window.ndim != 3 or window.shape[1:] != (543, 3):
        raise ValueError(f"expected a (frames, 543, 3) landmark window, got {window.shape}")
    return unit(_get_embedder().parts(window).mean(axis=0))


# --------------------------------------------------------------------------
# prototype store: a small JSON file {name: {"embedding": [384 floats], "clips": n}}
# --------------------------------------------------------------------------
_store_lock = threading.Lock()


def _read() -> dict:
    try:
        return json.loads(PROTOTYPES_PATH.read_text())
    except FileNotFoundError:
        return {}


def _write(store: dict) -> None:
    # write to a temp file then swap it in, so a crash mid-write can't corrupt the store
    tmp = PROTOTYPES_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(store))
    os.replace(tmp, PROTOTYPES_PATH)


def clean_name(name: str) -> str:
    """Lower-case + single spaces, so 'Thank  You' and 'thank you' are the same sign."""
    name = re.sub(r"\s+", " ", name).strip().lower()
    if not 1 <= len(name) <= 40:
        raise ValueError("sign name must be 1-40 characters")
    return name


def add_sign(name: str, windows: list[np.ndarray]) -> dict:
    """Teach a sign: average the embeddings of 3-5 clips into one prototype and save it.
    Teaching an existing name replaces it (re-teach)."""
    name = clean_name(name)
    if not FEWSHOT_MIN_CLIPS <= len(windows) <= FEWSHOT_MAX_CLIPS:
        raise ValueError(f"need {FEWSHOT_MIN_CLIPS}-{FEWSHOT_MAX_CLIPS} clips, got {len(windows)}")
    prototype = prototype_of(windows)
    with _store_lock:
        store = _read()
        store[name] = {"embedding": prototype.tolist(), "clips": len(windows)}
        _write(store)
    save_eval_clips(name, windows)
    return {"name": name, "clips": len(windows)}


def prototype_of(windows: list[np.ndarray]) -> np.ndarray:
    """Average the clips' embeddings into one unit vector: the sign's "average direction"."""
    return unit(np.stack([get_embedding(w) for w in windows]).mean(axis=0))


def similar_taught(windows: list[np.ndarray], name: str) -> list[tuple[str, float]]:
    """Already-taught signs (other than `name`) that the new clips resemble, as
    (name, cosine similarity), most similar first. "Resemble" = similarity of the new
    prototype to theirs is at least FEWSHOT_THRESHOLD, the same bar used to recognise."""
    new = prototype_of(windows)
    with _store_lock:
        store = _read()
    found = [(n, float(np.dot(new, np.array(s["embedding"], dtype=np.float32))))
             for n, s in store.items() if n != clean_name(name)]
    return sorted([f for f in found if f[1] >= FEWSHOT_THRESHOLD], key=lambda f: -f[1])


def is_taught(name: str) -> bool:
    with _store_lock:
        return clean_name(name) in _read()


def save_eval_clips(name: str, windows: list[np.ndarray]) -> None:
    """Keep each recorded clip as backend/data/eval_clips/<sign>/clip_<time>_<n>.npy so
    accuracy can be tested later. Teaching the same sign again adds to the folder; only
    the newest FEWSHOT_EVAL_MAX_CLIPS are kept. The folder name is made filesystem-safe."""
    folder = FEWSHOT_EVAL_DIR / (re.sub(r"[^a-z0-9]+", "_", clean_name(name)).strip("_") or "sign")
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for i, window in enumerate(windows, 1):
        np.save(folder / f"clip_{stamp}_{i}.npy", window.astype(np.float32))
    for old in sorted(folder.glob("clip_*.npy"))[:-FEWSHOT_EVAL_MAX_CLIPS]:
        old.unlink()


def best_match(embedding: np.ndarray) -> tuple[str, float] | None:
    """Closest taught sign as (name, cosine similarity) WHATEVER the similarity,
    or None if nothing is taught. The Teach tab shows this number so the
    threshold can be tuned by eye."""
    with _store_lock:
        store = _read()
    if not store:
        return None
    names = list(store)
    prototypes = np.array([store[n]["embedding"] for n in names], dtype=np.float32)
    sims = prototypes @ embedding          # both unit length -> dot product = cosine
    best = int(np.argmax(sims))
    return names[best], float(sims[best])


def match(embedding: np.ndarray, threshold: float = FEWSHOT_THRESHOLD) -> tuple[str, float] | None:
    """Best-matching taught sign as (name, cosine similarity), or None if nothing is
    taught or the best one is below the threshold (config.FEWSHOT_THRESHOLD)."""
    best = best_match(embedding)
    return best if best is not None and best[1] >= threshold else None


def hand_fraction(window: np.ndarray) -> float:
    """Share of frames in a (45, 543, 3) window where at least one hand was seen.
    Landmark layout: face 0-467, left hand 468-488, pose 489-521, right hand 522-542;
    undetected parts are NaN. Used to reject a clip recorded with no hands in view."""
    left = ~np.isnan(window[:, 468:489, 0]).any(axis=1)
    right = ~np.isnan(window[:, 522:543, 0]).any(axis=1)
    return float((left | right).mean())


def has_signs() -> bool:
    """Cheap check used on every live window: is anything taught at all?"""
    with _store_lock:
        return bool(_read())


def list_signs() -> list[dict]:
    with _store_lock:
        store = _read()
    return [{"name": n, "clips": store[n]["clips"]} for n in sorted(store)]


def delete_sign(name: str) -> bool:
    """Remove a taught sign. Returns False if there was no such sign."""
    name = clean_name(name)
    with _store_lock:
        store = _read()
        if name not in store:
            return False
        del store[name]
        _write(store)
    return True
