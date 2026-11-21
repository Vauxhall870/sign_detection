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
import json
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

from config import DATA_DIR, ISLR_LABELS_PATH
from ml.fewshot import Embedder, track_to_window, unit

RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "embedding_cache"   # landmark windows, so re-runs are instant


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
    """Run MediaPipe on every frame, then pick FEWSHOT_WINDOW_FRAMES (45) evenly spaced frames.

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

    window = track_to_window(np.stack(frames))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, window)
    return window


# --------------------------------------------------------------------------
# 2. window -> embedding (Embedder lives in ml/fewshot.py, shared with the app)
# --------------------------------------------------------------------------

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


def group_report(emb: np.ndarray, labels: np.ndarray, builtin: set, trials: int, rng):
    """Same experiment as few_shot_accuracy, but accuracy is reported separately for
    signs the pretrained model already knows ("built-in") and ones it never saw
    ("new") - new signs are what the Teach feature is actually for."""
    signs = sorted(set(labels))
    is_builtin = np.array([s in builtin for s in labels])
    print(f"  built-in signs: {[s for s in signs if s in builtin]}")
    print(f"  new signs     : {[s for s in signs if s not in builtin]}")
    for shots in (1, 3):
        correct = {"built-in": [], "new": []}
        for _ in range(trials):
            protos, held_out = [], np.ones(len(labels), dtype=bool)
            for s in signs:
                chosen = rng.choice(np.flatnonzero(labels == s), size=shots, replace=False)
                held_out[chosen] = False
                protos.append(unit(emb[chosen].mean(axis=0)))
            pred = np.array(signs)[(emb[held_out] @ np.stack(protos).T).argmax(axis=1)]
            hit = pred == labels[held_out]
            correct["built-in"].append(hit[is_builtin[held_out]].mean())
            correct["new"].append(hit[~is_builtin[held_out]].mean())
        print(f"  {shots}-shot accuracy   built-in {np.mean(correct['built-in']):.1%}   new {np.mean(correct['new']):.1%}")


def threshold_report(emb: np.ndarray, labels: np.ndarray, builtin: set, shots: int, trials: int, rng):
    """Help pick the match threshold. Simulates the app: ONLY the new signs are taught
    (prototype = average of `shots` clips). Then score
      - positives: held-out clips of a taught sign vs. their own prototype
      - negatives: clips of built-in signs vs. their best-matching taught prototype
    A good threshold accepts most positives and rejects most negatives."""
    new_signs = sorted(s for s in set(labels) if s not in builtin)
    pos, neg = [], []
    for _ in range(trials):
        protos, held_out = [], np.ones(len(labels), dtype=bool)
        for s in new_signs:
            chosen = rng.choice(np.flatnonzero(labels == s), size=shots, replace=False)
            held_out[chosen] = False
            protos.append(unit(emb[chosen].mean(axis=0)))
        protos = np.stack(protos)
        for i in np.flatnonzero(held_out):
            sims = emb[i] @ protos.T
            if labels[i] in new_signs:
                pos.append(sims[new_signs.index(labels[i])])
            else:
                neg.append(sims.max())
    pos, neg = np.array(pos), np.array(neg)
    pct = lambda a: " ".join(f"{np.percentile(a, p):.2f}" for p in (5, 25, 50, 75, 95))
    print(f"  {shots}-shot, taught = {new_signs}")
    print(f"  score percentiles (5/25/50/75/95)  taught clip vs own prototype: {pct(pos)}")
    print(f"                                     built-in clip vs best prototype: {pct(neg)}")
    print(f"  {'threshold':>9}{'taught accepted':>17}{'built-in wrongly accepted':>27}")
    for t in np.arange(0.10, 0.61, 0.05):
        print(f"  {t:>9.2f}{(pos >= t).mean():>17.1%}{(neg >= t).mean():>27.1%}")


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

    # ---- per-group split, on option (b) (the one we use) ----
    # signs the pretrained model was trained on = keys of its label map
    builtin = set(json.loads(ISLR_LABELS_PATH.read_text()))
    emb_b = options["(b) average of 3 x 384 (384-d)"]
    print("\n== per-group accuracy, option (b) ==")
    group_report(emb_b, labels, builtin, args.trials, rng)

    is_new = np.array([s not in builtin for s in labels])
    same_new, diff_new = similarity_report(emb_b[is_new], labels[is_new])
    same_all, diff_all = similarity_report(emb_b, labels)
    print("\n== same-sign similarity by group, option (b) ==")
    print(f"  new signs only : same {same_new:.3f}   diff {diff_new:.3f}   gap {same_new - diff_new:.3f}")
    print(f"  all signs      : same {same_all:.3f}   diff {diff_all:.3f}   gap {same_all - diff_all:.3f}")

    print("\n== threshold sweep, option (b) ==")
    threshold_report(emb_b, labels, builtin, shots=3, trials=args.trials, rng=rng)


if __name__ == "__main__":
    sys.exit(main())
