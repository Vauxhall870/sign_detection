"""Sanity check for few-shot sign learning: are the ISLR model's embeddings useful?

Idea: the pretrained model squeezes a sign clip into a vector (an "embedding")
just before its final 250-way classifier. If that vector captures *what the
sign is*, clips of the SAME sign should point in nearly the same direction
(high cosine similarity) and clips of DIFFERENT signs should not. If that holds,
we can teach the app a new sign by storing the average embedding of a few clips
(a "prototype") and matching new clips against it - no retraining needed.

This script measures that on recorded clips in data/raw/<sign>/*.mp4 and compares
two ways of building one vector out of the model's three sub-networks:
  (a) concatenation of the three 384-d embeddings, each L2-normalised first -> 1152-d
  (b) the plain average of the three 384-d embeddings                        -> 384-d

Run (from backend/):   python -m ml.check_embeddings
"""
import argparse
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

from config import DATA_DIR, ISLR_MODEL_PATH

RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "embedding_cache"   # landmark windows, so re-runs are instant
WINDOW_FRAMES = 45                         # what the live app feeds the model (30 fps x 1.5 s)


# --------------------------------------------------------------------------
# 1. clip -> (45, 543, 3) landmark window
# --------------------------------------------------------------------------
_holistic = None


def _init_worker():
    """Each worker process needs its own MediaPipe object (not picklable/thread-safe)."""
    global _holistic
    from ml.landmarks import make_holistic
    _holistic = make_holistic(static=False)


def clip_to_window(path: Path) -> np.ndarray:
    """Run MediaPipe on every frame, then pick 45 evenly spaced frames.

    A recorded clip is one whole sign, so we stretch it over the fixed 45-frame
    window the model expects - the same shape the live app builds from the webcam.
    """
    import cv2
    from ml.landmarks import detect, extract_raw543

    cache = CACHE_DIR / f"{path.parent.name}__{path.stem}.npy"
    if cache.exists():
        return np.load(cache)

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(extract_raw543(detect(frame, _holistic)))   # (543, 3), NaN = not seen
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {path}")

    track = np.stack(frames).astype(np.float32)
    idx = np.linspace(0, len(track) - 1, WINDOW_FRAMES).round().astype(int)
    window = track[idx]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, window)
    return window


# --------------------------------------------------------------------------
# 2. window -> embedding (own interpreter; the app's live one is untouched)
# --------------------------------------------------------------------------
class Embedder:
    """Runs the ISLR TFLite model and reads the tensors just before each classifier."""

    def __init__(self):
        import tensorflow as tf

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

    def parts(self, window: np.ndarray) -> np.ndarray:
        """-> (3, 384): one raw embedding per sub-network."""
        self._run(inputs=window.astype(np.float32))
        return np.stack([self._it.get_tensor(i)[0] for i in self._pool_ids])


def unit(v: np.ndarray) -> np.ndarray:
    """Scale to length 1, so a dot product between two vectors IS their cosine similarity."""
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def variant_a(parts: np.ndarray) -> np.ndarray:
    """(a) L2-normalise each 384-d part, concatenate to 1152-d, normalise again."""
    return unit(np.concatenate(unit(parts)))


def variant_b(parts: np.ndarray) -> np.ndarray:
    """(b) average the three 384-d parts, normalise."""
    return unit(parts.mean(axis=0))


# --------------------------------------------------------------------------
# 3. metrics
# --------------------------------------------------------------------------
def similarity_report(emb: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Mean cosine similarity over same-sign pairs and over different-sign pairs."""
    sim = emb @ emb.T
    same = labels[:, None] == labels[None, :]
    off_diag = ~np.eye(len(labels), dtype=bool)          # a clip is trivially 1.0 with itself
    return float(sim[same & off_diag].mean()), float(sim[~same].mean())


def few_shot_accuracy(emb: np.ndarray, labels: np.ndarray, shots: int, trials: int, rng) -> float:
    """Simulate teaching: per sign pick `shots` clips as the taught examples,
    average them into a prototype, then classify every OTHER clip by the most
    similar prototype. Repeated with random picks; returns mean accuracy."""
    signs = sorted(set(labels))
    accs = []
    for _ in range(trials):
        protos, held_out = [], np.ones(len(labels), dtype=bool)
        for s in signs:
            members = np.flatnonzero(labels == s)
            chosen = rng.choice(members, size=shots, replace=False)
            held_out[chosen] = False
            protos.append(unit(emb[chosen].mean(axis=0)))
        protos = np.stack(protos)
        pred = np.array(signs)[(emb[held_out] @ protos.T).argmax(axis=1)]
        accs.append(float((pred == labels[held_out]).mean()))
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signs", type=int, default=8, help="how many signs to use")
    ap.add_argument("--clips", type=int, default=8, help="clips per sign")
    ap.add_argument("--trials", type=int, default=300, help="random prototype draws")
    args = ap.parse_args()

    sign_dirs = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())[: args.signs]
    paths, labels = [], []
    for d in sign_dirs:
        clips = sorted(d.glob("*.mp4"))[: args.clips]
        paths += clips
        labels += [d.name] * len(clips)
    labels = np.array(labels)
    print(f"{len(sign_dirs)} signs x up to {args.clips} clips = {len(paths)} clips: "
          f"{', '.join(d.name for d in sign_dirs)}")

    print(f"extracting landmarks on {min(cpu_count(), 12)} processes (cached after the first run)...")
    with Pool(min(cpu_count(), 12), initializer=_init_worker) as pool:
        windows = pool.map(clip_to_window, paths, chunksize=1)

    embedder = Embedder()
    parts = [embedder.parts(w) for w in windows]
    options = {
        "(a) concat of 3 x 384, each L2-normed (1152-d)": np.stack([variant_a(p) for p in parts]),
        "(b) average of 3 x 384 (384-d)": np.stack([variant_b(p) for p in parts]),
    }

    rng = np.random.default_rng(0)
    chance = 1 / len(sign_dirs)
    print(f"\nchance accuracy = 1/{len(sign_dirs)} = {chance:.1%}\n")
    print(f"{'option':<48}{'same':>8}{'diff':>8}{'gap':>8}{'1-shot':>9}{'3-shot':>9}")
    for name, emb in options.items():
        same, diff = similarity_report(emb, labels)
        one = few_shot_accuracy(emb, labels, 1, args.trials, rng)
        three = few_shot_accuracy(emb, labels, 3, args.trials, rng)
        print(f"{name:<48}{same:>8.3f}{diff:>8.3f}{same - diff:>8.3f}{one:>9.1%}{three:>9.1%}")
    print("\nsame = mean cosine similarity of clips of the same sign; diff = of different signs;"
          "\ngap = same - diff (bigger is better); n-shot = accuracy classifying held-out clips"
          "\nfrom a prototype averaged over n taught clips per sign.")


if __name__ == "__main__":
    sys.exit(main())
