"""Evaluate few-shot sign teaching (the Teach tab) - accuracy, threshold, baseline, speed.

    python -m ml.evaluate                    # everything (a few minutes + the LSTM baseline)
    python -m ml.evaluate --lstm-episodes 0  # skip the slow LSTM baseline

This only READS data and writes backend/results/; it never touches the live app,
data/prototypes.json or data/eval_clips/.

Two groups of signs are evaluated separately:
  ISL    the 10 signs taught through the Teach tab (data/eval_clips/<sign>/*.npy, 8 clips
         each, one signer, one session)
  WLASL  the signs in data/raw that the pretrained model has NEVER seen ("new" signs):
         many different signers, but each sign has only ~13-16 clips

Experiments (each writes a CSV and a plot):
  1. K-shot accuracy   K = 1, 3, 5 taught clips per sign. ISL = 10-way, WLASL = 5-way.
  2. LSTM baseline     the project's LSTM, trained from scratch on the SAME K clips per sign
                       and tested on the SAME query clips (slow, so fewer episodes).
  3. Threshold sweep   0.20 .. 0.50: share of correct signs accepted vs wrong signs accepted.
  4. Speed             milliseconds per window in ISL mode.
  5. Confusion matrix  ISL at K = 3: which signs get mixed up.

How an "episode" works (it simulates teaching, then testing):
  - pick the signs (ISL: all 10; WLASL: 5 random of the 7 new ones)
  - for each sign pick K random clips as the TEACHING clips -> average their embeddings
    into a prototype (exactly what Save does)
  - every OTHER clip of those signs is a TEST clip. A clip is never both taught and tested
    inside one episode.
  - classify each test clip by its most similar prototype (cosine similarity)
  Accuracy of the episode = share of test clips classified correctly. We run 400 episodes with a
  fixed seed and report the mean and a 95% confidence interval over episodes.
"""
import argparse
import csv
import json
import tempfile
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from scipy import stats

from config import DATA_DIR, FEWSHOT_EVAL_DIR, FEWSHOT_THRESHOLD, ISLR_LABELS_PATH, ISLR_MODEL_PATH, BASE_DIR
from ml import fewshot
from ml.fewshot import unit

RESULTS_DIR = BASE_DIR / "results"
RAW_DIR = DATA_DIR / "raw"
SEED = 0
KS = (1, 3, 5)
THRESHOLDS = np.round(np.arange(0.20, 0.5001, 0.025), 3)   # 0.20 ... 0.50


# --------------------------------------------------------------------------
# 1. Load data: every clip as a (45, 543, 3) landmark window, then as an embedding
# --------------------------------------------------------------------------
def load_isl_windows() -> dict[str, list[np.ndarray]]:
    """The clips saved by the Teach tab: data/eval_clips/<sign>/clip_*.npy"""
    return {d.name: [np.load(f) for f in sorted(d.glob("clip_*.npy"))]
            for d in sorted(FEWSHOT_EVAL_DIR.iterdir()) if d.is_dir()}


def load_wlasl_windows() -> dict[str, list[np.ndarray]]:
    """WLASL videos in data/raw/<sign>/*.mp4 -> landmark windows (cached on disk by
    check_embeddings, so only the first run pays for MediaPipe)."""
    from ml.check_embeddings import _init_worker, clip_to_window   # reuse the exact same extraction
    paths = [p for d in sorted(RAW_DIR.iterdir()) if d.is_dir() for p in sorted(d.glob("*.mp4"))]
    print(f"WLASL: {len(paths)} clips; extracting landmarks for the uncached ones "
          f"on {min(cpu_count(), 12)} processes...")
    with Pool(min(cpu_count(), 12), initializer=_init_worker) as pool:
        windows = pool.map(clip_to_window, paths, chunksize=1)
    out: dict[str, list] = {}
    for p, w in zip(paths, windows):
        out.setdefault(p.parent.name, []).append(w)
    return out


def embed_all(groups: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    """sign -> (n_clips, 384) unit embeddings, using the same get_embedding as the live app."""
    return {s: np.stack([fewshot.get_embedding(w) for w in ws]) for s, ws in groups.items()}


# --------------------------------------------------------------------------
# 2. Episodes
# --------------------------------------------------------------------------
class Episode:
    """One simulated teaching session: which signs, which clips were taught, which are tested."""

    def __init__(self, signs, support, queries):
        self.signs = signs        # list of sign names, in prototype order
        self.support = support    # sign -> array of K clip indices (the teaching clips)
        self.queries = queries    # sign -> array of the remaining clip indices (the test clips)


def make_episodes(counts: dict[str, int], n_way: int | None, k: int, n: int, seed) -> list[Episode]:
    """`counts` = clips available per sign. n_way=None uses every sign."""
    rng = np.random.default_rng(seed)
    names = sorted(counts)
    episodes = []
    for _ in range(n):
        signs = names if n_way is None else sorted(rng.choice(names, size=n_way, replace=False))
        support, queries = {}, {}
        for s in signs:
            order = rng.permutation(counts[s])
            support[s], queries[s] = order[:k], order[k:]   # test clips = everything NOT taught
        episodes.append(Episode(list(signs), support, queries))
    return episodes


def prototypes(emb: dict[str, np.ndarray], ep: Episode) -> np.ndarray:
    """(n_signs, 384): the average of each sign's K teaching embeddings, normalised."""
    return np.stack([unit(emb[s][ep.support[s]].mean(axis=0)) for s in ep.signs])


def run_prototype_episode(emb, ep):
    """-> (true_idx, predicted_idx, best_similarity) for every test clip of one episode."""
    protos = prototypes(emb, ep)
    true, pred, sim = [], [], []
    for i, s in enumerate(ep.signs):
        sims = emb[s][ep.queries[s]] @ protos.T            # (n_test, n_signs) cosine similarities
        true += [i] * len(sims)
        pred += list(sims.argmax(axis=1))
        sim += list(sims.max(axis=1))
    return np.array(true), np.array(pred), np.array(sim)


def mean_ci(values) -> tuple[float, float]:
    """Mean and the half-width of its 95% confidence interval (t distribution)."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float(v.mean()), float("nan")
    return float(v.mean()), float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


# --------------------------------------------------------------------------
# 3. LSTM baseline: the project's LSTM (ml/train.py) trained on the same K clips
# --------------------------------------------------------------------------
def window_to_lstm(window: np.ndarray, idx=None) -> np.ndarray:
    """(45, 543, 3) Kaggle-layout window -> (30, 258) LSTM input, the same feature layout as
    ml.landmarks.extract_keypoints: pose(33 x [x,y,z,visibility]) + left hand + right hand,
    all expressed in the signer's shoulder frame, zeros for parts not seen.
    Difference: the saved windows have no pose visibility score, so it is set to 1 whenever
    the pose was detected."""
    if idx is None:
        idx = np.linspace(0, len(window) - 1, 30).round().astype(int)   # 45 frames -> 30
    out = np.zeros((len(idx), 258), dtype=np.float32)
    for row, f in enumerate(idx):
        lh, pose, rh = window[f, 468:489], window[f, 489:522], window[f, 522:543]
        if not np.isnan(pose).any():
            origin = (pose[11] + pose[12]) / 2.0                          # mid-shoulder
            scale = float(np.linalg.norm(pose[11, :2] - pose[12, :2]))
            scale = scale if scale >= 1e-3 else 1.0
            pose, lh, rh = (pose - origin) / scale, (lh - origin) / scale, (rh - origin) / scale
            vis = np.ones((33, 1), dtype=np.float32)
        else:
            pose, vis = np.zeros((33, 3), np.float32), np.zeros((33, 1), np.float32)
        lh = np.zeros((21, 3), np.float32) if np.isnan(lh).any() else lh
        rh = np.zeros((21, 3), np.float32) if np.isnan(rh).any() else rh
        out[row] = np.concatenate([np.concatenate([pose, vis], axis=1).ravel(), lh.ravel(), rh.ravel()])
    return np.nan_to_num(out)


def lstm_episode(windows: dict[str, list], ep: Episode, seed: int, epochs: int, copies: int = 8):
    """Train a fresh LSTM on the K teaching clips of each sign, then classify the test clips.
    Each teaching clip is also used with `copies` random temporal subsamples (like the
    temporal jitter ml.preprocess uses) because a handful of sequences would otherwise be
    hopeless. Same test clips as the prototype method. -> (true_idx, predicted_idx)."""
    from tensorflow import keras
    from ml.train import build_model
    rng = np.random.default_rng(seed)
    keras.utils.set_random_seed(seed)
    X, y = [], []
    for i, s in enumerate(ep.signs):
        for c in ep.support[s]:
            w = windows[s][c]
            X.append(window_to_lstm(w)); y.append(i)
            for _ in range(copies):
                idx = np.sort(rng.choice(len(w), size=30, replace=False))   # random 30 of 45 frames
                X.append(window_to_lstm(w, idx)); y.append(i)
    model = build_model(len(ep.signs))
    model.fit(np.stack(X), np.array(y), epochs=epochs, batch_size=16, verbose=0)
    true, Xq = [], []
    for i, s in enumerate(ep.signs):
        for c in ep.queries[s]:
            Xq.append(window_to_lstm(windows[s][c])); true.append(i)
    pred = model.predict(np.stack(Xq), verbose=0).argmax(axis=1)
    keras.backend.clear_session()
    return np.array(true), pred


# --------------------------------------------------------------------------
# 4. Experiments
# --------------------------------------------------------------------------
def exp_kshot(datasets, n_episodes, lstm_episodes, lstm_epochs):
    """Experiment 1 + 2. Returns (summary rows, per-episode rows, confusion data for ISL K=3)."""
    summary, per_episode, confusion = [], [], None
    for ds_id, (name, emb, windows, n_way) in enumerate(datasets):
        counts = {s: len(v) for s, v in emb.items()}
        chance = 1 / (n_way or len(counts))
        for k in KS:
            episodes = make_episodes(counts, n_way, k, n_episodes, seed=[SEED, ds_id, k])
            accs, conf = [], None
            for ep in episodes:
                true, pred, _ = run_prototype_episode(emb, ep)
                accs.append(float((true == pred).mean()))
                if name == "ISL" and k == 3:           # aggregate for the confusion matrix
                    conf = np.zeros((len(ep.signs),) * 2) if conf is None else conf
                    np.add.at(conf, (true, pred), 1)
            if conf is not None:
                confusion = (ep.signs, conf)
            m, ci = mean_ci(accs)
            summary.append(dict(group=name, way=n_way or len(counts), k=k, method="prototype (ours)",
                                episodes=len(accs), mean=m, ci95=ci, chance=chance))
            per_episode += [dict(group=name, k=k, method="prototype (ours)", episode=i, accuracy=a)
                            for i, a in enumerate(accs)]
            print(f"  {name:5s} K={k}  prototype  {m:6.1%} ± {ci:.1%}  ({len(accs)} episodes, chance {chance:.0%})")

            # ---- baseline on the FIRST few of the very same episodes ----
            if lstm_episodes > 0:
                sub = episodes[:lstm_episodes]
                t0, l_accs = time.time(), []
                for j, ep in enumerate(sub):
                    true, pred = lstm_episode(windows, ep, seed=1000 * k + j, epochs=lstm_epochs)
                    l_accs.append(float((true == pred).mean()))
                lm, lci = mean_ci(l_accs)
                pm, pci = mean_ci(accs[:lstm_episodes])      # prototype on the same subset: fair comparison
                summary.append(dict(group=name, way=n_way or len(counts), k=k, method="LSTM from scratch",
                                    episodes=len(l_accs), mean=lm, ci95=lci, chance=chance))
                summary.append(dict(group=name, way=n_way or len(counts), k=k, method="prototype (same episodes as LSTM)",
                                    episodes=len(l_accs), mean=pm, ci95=pci, chance=chance))
                per_episode += [dict(group=name, k=k, method="LSTM from scratch", episode=i, accuracy=a)
                                for i, a in enumerate(l_accs)]
                print(f"  {name:5s} K={k}  LSTM       {lm:6.1%} ± {lci:.1%}  ({len(l_accs)} episodes, "
                      f"{time.time() - t0:.0f}s; prototype on same episodes {pm:.1%})")
    return summary, per_episode, confusion


def exp_threshold(isl_emb, wlasl_emb, n_episodes, k=3):
    """Experiment 3. For each threshold t, using K=3 taught clips per sign:
         correct accepted = share of test clips of TAUGHT signs whose closest prototype is the
                            right sign AND whose similarity >= t   (the app would show the right word)
         wrong-label      = share of those whose similarity >= t but closest prototype is wrong
         wrong accepted   = share of clips of signs that were NOT taught whose best similarity >= t
                            (the app would show some taught word for a sign it was never taught)
       'Not taught' clips come from (a) a sign held out of teaching (same signer/session) and
       (b) the WLASL clips, i.e. other signers on other cameras."""
    rows = []
    rng = np.random.default_rng([SEED, 99])
    pools = {}                                          # name -> lists of (similarity, correct) / negatives
    # (a) ISL: hold one sign out of teaching each episode; teach the other 9
    # (b) WLASL: teach 5 of the 7 new signs, the other 2 are 'not taught'
    for name, emb, n_taught in (("ISL", isl_emb, len(isl_emb) - 1), ("WLASL", wlasl_emb, 5)):
        names = sorted(emb)
        pos_sim, pos_ok, neg_sim = [], [], []
        for _ in range(n_episodes):
            taught = sorted(rng.choice(names, size=n_taught, replace=False))
            unknown = [s for s in names if s not in taught]
            support, queries = {}, {}
            for s in taught:
                order = rng.permutation(len(emb[s]))
                support[s], queries[s] = order[:k], order[k:]
            ep = Episode(taught, support, queries)
            true, pred, sim = run_prototype_episode(emb, ep)
            pos_sim += list(sim); pos_ok += list(true == pred)
            protos = prototypes(emb, ep)
            for s in unknown:
                neg_sim += list((emb[s] @ protos.T).max(axis=1))
        pools[name] = (np.array(pos_sim), np.array(pos_ok), np.array(neg_sim))
    # (c) ISL taught, WLASL clips (other signers / camera) as the 'not taught' signs
    pos_sim, pos_ok, neg_sim = [], [], []
    wl_all = np.concatenate([wlasl_emb[s] for s in sorted(wlasl_emb)])
    for _ in range(n_episodes):
        ep = make_episodes({s: len(v) for s, v in isl_emb.items()}, None, k, 1, seed=int(rng.integers(1 << 31)))[0]
        true, pred, sim = run_prototype_episode(isl_emb, ep)
        pos_sim += list(sim); pos_ok += list(true == pred)
        neg_sim += list((wl_all @ prototypes(isl_emb, ep).T).max(axis=1))
    pools["ISL vs WLASL clips"] = (np.array(pos_sim), np.array(pos_ok), np.array(neg_sim))

    labels = {"ISL": "ISL: not-taught = one ISL sign held out",
              "WLASL": "WLASL: not-taught = the 2 unused new signs",
              "ISL vs WLASL clips": "ISL taught: not-taught = WLASL clips (other signers)"}
    for name, (ps, ok, ns) in pools.items():
        for t in THRESHOLDS:
            rows.append(dict(setting=labels[name], threshold=float(t),
                             correct_accepted=float(((ps >= t) & ok).mean()),
                             wrong_label_accepted=float(((ps >= t) & ~ok).mean()),
                             not_taught_accepted=float((ns >= t).mean()),
                             n_taught_clips=len(ps), n_not_taught_clips=len(ns)))
    return rows


def exp_speed(isl_windows, wlasl_frames_path):
    """Experiment 4: milliseconds per window in ISL mode, on this machine's CPU.
    ISL mode per frame = MediaPipe Holistic + embedding + matching (the built-in model is not run).
    We time the real functions the server calls, with a temporary prototypes file holding the
    10 ISL signs, so the live data/prototypes.json is untouched."""
    import cv2
    from ml.landmarks import detect, extract_raw543, make_holistic
    rows = []

    def timed(fn, items, warmup=10):
        for it in items[:warmup]:
            fn(it)
        ts = []
        for it in items:
            t0 = time.perf_counter(); fn(it); ts.append((time.perf_counter() - t0) * 1000)
        return np.array(ts)

    windows = [w for ws in isl_windows.values() for w in ws]
    with tempfile.TemporaryDirectory() as tmp:
        # point the store at a throw-away file (module variable; nothing else sees this change)
        fewshot.PROTOTYPES_PATH = Path(tmp) / "p.json"
        store = {s: {"embedding": fewshot.unit(np.stack([fewshot.get_embedding(w) for w in ws]).mean(axis=0)).tolist(),
                     "clips": len(ws)} for s, ws in isl_windows.items()}
        fewshot._write(store)
        embed = timed(fewshot.get_embedding, windows * 3)                           # TFLite embedding only
        embs = [fewshot.get_embedding(w) for w in windows]
        match = timed(fewshot.best_match, embs * 3)                                 # read store + cosine match
        isl_window = timed(lambda w: fewshot.best_match(fewshot.get_embedding(w)), windows * 3)  # = ISL decision per window
        from ml.predict import recognizer
        recognizer.load()
        asl_window = timed(recognizer.predict, windows * 3)                         # built-in 250-sign model, for comparison

        # MediaPipe Holistic per frame on a real video (the dominant cost in the live app)
        holistic = make_holistic()
        cap, frames = cv2.VideoCapture(str(wlasl_frames_path)), []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        mp_ms = timed(lambda f: extract_raw543(detect(f, holistic)), frames * 3, warmup=5)
        holistic.close()
        fewshot.PROTOTYPES_PATH = fewshot.FEWSHOT_PROTOTYPES_PATH          # restore

    for name, ts in (("embedding only (TFLite)", embed), ("matching 10 prototypes (incl. reading the store)", match),
                     ("ISL decision per window (embedding + matching)", isl_window),
                     ("ASL decision per window (built-in 250-sign model), for comparison", asl_window),
                     ("MediaPipe Holistic per frame", mp_ms)):
        rows.append(dict(step=name, n=len(ts), mean_ms=ts.mean(), median_ms=np.median(ts),
                         p95_ms=np.percentile(ts, 95), max_ms=ts.max()))
    total = mp_ms.mean() + isl_window.mean()
    rows.append(dict(step="ISL mode total per frame (Holistic + decision)", n=0, mean_ms=total,
                     median_ms=np.median(mp_ms) + np.median(isl_window),
                     p95_ms=np.percentile(mp_ms, 95) + np.percentile(isl_window, 95), max_ms=float("nan")))
    return rows


# --------------------------------------------------------------------------
# 5. Output: CSVs, plots, results.md
# --------------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def make_plots(summary, thr_rows, confusion, speed_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- K-shot accuracy ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    styles = {"prototype (ours)": ("tab:blue", "o-"), "LSTM from scratch": ("tab:red", "s--")}
    for ax, group in zip(axes, ("ISL", "WLASL")):
        rows = [r for r in summary if r["group"] == group]
        for method, (color, fmt) in styles.items():
            pts = sorted((r["k"], r["mean"] * 100, r["ci95"] * 100) for r in rows if r["method"] == method)
            if pts:
                ks, ms, cis = zip(*pts)
                ax.errorbar(ks, ms, yerr=cis, fmt=fmt, color=color, capsize=4, label=method)
        chance = rows[0]["chance"] * 100
        ax.axhline(chance, color="gray", ls=":", label=f"chance ({chance:.0f}%)")
        ax.set_title(f"{group}: {rows[0]['way']}-way few-shot accuracy"); ax.set_xticks(KS)
        ax.set_xlabel("K = taught clips per sign"); ax.set_ylim(0, 105); ax.grid(alpha=.3)
    axes[0].set_ylabel("accuracy on test clips (%)  [mean ± 95% CI]"); axes[0].legend(loc="lower right")
    fig.suptitle("Experiment 1-2: K-shot accuracy, prototype method vs LSTM baseline"); fig.tight_layout()
    fig.savefig(RESULTS_DIR / "kshot_accuracy.png", dpi=150); plt.close(fig)

    # ---- threshold sweep ----
    settings = sorted({r["setting"] for r in thr_rows})
    fig, axes = plt.subplots(1, len(settings), figsize=(5 * len(settings), 4.2), sharey=True)
    for ax, s in zip(np.atleast_1d(axes), settings):
        rs = [r for r in thr_rows if r["setting"] == s]
        t = [r["threshold"] for r in rs]
        ax.plot(t, [r["correct_accepted"] * 100 for r in rs], "o-", color="tab:green", label="correct sign accepted")
        ax.plot(t, [r["not_taught_accepted"] * 100 for r in rs], "s-", color="tab:red", label="not-taught sign wrongly accepted")
        ax.axvline(FEWSHOT_THRESHOLD, color="k", ls=":", label=f"current threshold ({FEWSHOT_THRESHOLD})")
        ax.set_title(s, fontsize=9); ax.set_xlabel("similarity threshold"); ax.grid(alpha=.3)
    np.atleast_1d(axes)[0].set_ylabel("% of clips accepted"); np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle("Experiment 3: threshold sweep (K = 3)"); fig.tight_layout()
    fig.savefig(RESULTS_DIR / "threshold_sweep.png", dpi=150); plt.close(fig)

    # ---- confusion matrix ----
    signs, conf = confusion
    pct = conf / conf.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(signs))); ax.set_xticklabels(signs, rotation=45, ha="right")
    ax.set_yticks(range(len(signs))); ax.set_yticklabels(signs)
    for i in range(len(signs)):
        for j in range(len(signs)):
            if pct[i, j] >= 0.5:
                ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if pct[i, j] > 55 else "black")
    ax.set_xlabel("predicted sign"); ax.set_ylabel("true sign")
    ax.set_title("Experiment 5: ISL confusion matrix, K = 3\n(% of test clips; each row sums to 100)")
    fig.colorbar(im, label="%"); fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_isl_k3.png", dpi=150); plt.close(fig)

    # ---- speed ----
    fig, ax = plt.subplots(figsize=(9, 3.8))
    labels = [r["step"] for r in speed_rows]
    ax.barh(labels, [r["mean_ms"] for r in speed_rows], color="tab:purple")
    ax.invert_yaxis(); ax.set_xlabel("milliseconds (mean)"); ax.tick_params(axis="y", labelsize=7)
    ax.set_title("Experiment 4: time per window / frame (this CPU)"); fig.tight_layout()
    fig.savefig(RESULTS_DIR / "speed.png", dpi=150); plt.close(fig)


def write_markdown(summary, thr_rows, confusion, speed_rows, args, n_clips):
    """results.md: the numbers (generated) plus the caveats (fixed text)."""
    f = lambda r: f"{r['mean'] * 100:.1f}% ± {r['ci95'] * 100:.1f}"
    L = ["# Few-shot sign teaching: evaluation results", "",
         f"Generated by `python -m ml.evaluate` (seed {SEED}). Plots and CSVs are in this folder.", "",
         f"Data: **ISL** = {n_clips['ISL']} clips (10 signs taught through the Teach tab, one signer, one session). "
         f"**WLASL new signs** = {n_clips['WLASL']} clips of 7 signs the pretrained model never saw, many signers.", "",
         "## 1-2. K-shot accuracy and LSTM baseline (`kshot_accuracy.png`, `kshot_accuracy.csv`)", "",
         "Mean accuracy % ± 95% CI over episodes. Teaching clips are never used as test clips. "
         f"Prototype = {args.episodes} episodes. The LSTM is slow to train so it ran on the first "
         f"{args.lstm_episodes} of those same episodes; the last column is the prototype method on exactly those episodes.", "",
         "| Group | K | Chance | Prototype (ours) | LSTM from scratch | Prototype on LSTM's episodes |", "|---|---|---|---|---|---|"]
    for g in ("ISL", "WLASL"):
        for k in KS:
            get = lambda m: next((r for r in summary if r["group"] == g and r["k"] == k and r["method"] == m), None)
            p, l, ps = get("prototype (ours)"), get("LSTM from scratch"), get("prototype (same episodes as LSTM)")
            L.append(f"| {g} ({p['way']}-way) | {k} | {p['chance'] * 100:.0f}% | {f(p)} | {f(l) if l else 'n/a'} | {f(ps) if ps else 'n/a'} |")
    L += ["", "## 3. Threshold sweep, K = 3 (`threshold_sweep.png`, `threshold_sweep.csv`)", "",
          "*Correct accepted* = the clip's closest prototype is the right sign AND similarity >= threshold. "
          "*Not-taught accepted* = a clip of a sign that was never taught still scores >= threshold against some taught sign "
          "(the app would wrongly show a taught word). The app uses 0.35.", "",
          "| Setting | Threshold | Correct accepted | Wrong label accepted | Not-taught accepted |", "|---|---|---|---|---|"]
    for r in thr_rows:
        if round(r["threshold"] * 40) % 2 == 0 or abs(r["threshold"] - FEWSHOT_THRESHOLD) < 1e-9:   # every 0.05, plus 0.35
            L.append(f"| {r['setting']} | {r['threshold']:.2f} | {r['correct_accepted'] * 100:.1f}% | "
                     f"{r['wrong_label_accepted'] * 100:.1f}% | {r['not_taught_accepted'] * 100:.1f}% |")
    L += ["", "## 4. Speed (`speed.csv`, `speed.png`)", "", "| Step | mean ms | median ms | p95 ms |", "|---|---|---|---|"]
    L += [f"| {r['step']} | {r['mean_ms']:.1f} | {r['median_ms']:.1f} | {r['p95_ms']:.1f} |" for r in speed_rows]
    signs, conf = confusion
    pct = conf / conf.sum(axis=1, keepdims=True) * 100
    pairs = sorted(((pct[i, j], signs[i], signs[j]) for i in range(len(signs)) for j in range(len(signs)) if i != j), reverse=True)[:6]
    L += ["", "## 5. ISL confusion matrix, K = 3 (`confusion_isl_k3.png`, `confusion_isl_k3.csv`)", "",
          "Most frequent mix-ups (true sign -> predicted sign, % of that sign's test clips):", ""]
    L += [f"- {a} -> {b}: {p:.1f}%" for p, a, b in pairs if p > 0] or ["- none"]
    L += ["", "## Limitations", "",
          "- **One signer, one session (ISL).** All ISL clips were recorded back to back by one person with one camera, "
          "so test clips are near-duplicates of the teaching clips. The ISL numbers are an upper bound for "
          "'the same person using the app right after teaching'; they say nothing about other signers.",
          "- **WLASL is the opposite problem:** many signers and cameras, but only 13-16 clips per sign and only 7 new signs, "
          "so the 5-way task reuses the same few signs in different combinations.",
          "- **Episodes are not independent.** They resample the same small clip pool, so the confidence intervals "
          "describe variation from *which clips were picked*, not from new signers or new days. Real uncertainty is larger.",
          "- **The LSTM baseline** is trained from scratch on K x signs sequences (plus 8 random temporal subsamples of each) "
          f"for {args.lstm_epochs} epochs, with no tuning, and on the first {args.lstm_episodes} episodes only. "
          "It is the project's architecture, not the strongest possible few-shot baseline. Its input is rebuilt from the saved "
          "543-landmark windows (pose visibility set to 1), not from raw video.",
          "- **Threshold sweep:** 'not taught' signs are only a few signs (one ISL sign per episode, 2 WLASL signs) so the "
          "false-accept rates are rough. WLASL-as-negatives also differs in camera and signer, which can make rejection look easier.",
          "- **Speed** is for this laptop CPU with nothing else running; the live app adds JPEG decoding and WebSocket time.",
          "- Mirrored (left-handed) signing, lighting changes, and signers who are not the teacher were not tested for ISL."]
    (RESULTS_DIR / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400, help="random episodes per setting (prototype method)")
    ap.add_argument("--lstm-episodes", type=int, default=30, help="how many of those episodes also train the LSTM (0 = skip)")
    ap.add_argument("--lstm-epochs", type=int, default=60)
    args = ap.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    print("loading clips...")
    isl_w = load_isl_windows()
    wl_all = load_wlasl_windows()
    builtin = set(json.loads(ISLR_LABELS_PATH.read_text()))         # signs the pretrained model was trained on
    wl_w = {s: w for s, w in wl_all.items() if s not in builtin}    # keep only the NEW signs
    print(f"ISL: {', '.join(f'{s}({len(v)})' for s, v in isl_w.items())}")
    print(f"WLASL new: {', '.join(f'{s}({len(v)})' for s, v in wl_w.items())}")
    isl_e, wl_e = embed_all(isl_w), embed_all(wl_w)
    wl_every = embed_all(wl_all)                                      # all WLASL clips, used as 'not taught' negatives

    print("\n[1-2] K-shot accuracy")
    summary, per_ep, confusion = exp_kshot(
        [("ISL", isl_e, isl_w, None), ("WLASL", wl_e, wl_w, 5)], args.episodes, args.lstm_episodes, args.lstm_epochs)
    write_csv(RESULTS_DIR / "kshot_accuracy.csv", summary)
    write_csv(RESULTS_DIR / "kshot_episodes.csv", per_ep)

    print("\n[3] threshold sweep")
    thr = exp_threshold(isl_e, wl_every, args.episodes)
    write_csv(RESULTS_DIR / "threshold_sweep.csv", thr)

    print("\n[4] speed")
    some_clip = next(iter(sorted((RAW_DIR / "drink").glob("*.mp4"))))
    speed = exp_speed(isl_w, some_clip)
    write_csv(RESULTS_DIR / "speed.csv", speed)
    for r in speed:
        print(f"  {r['step']:<70s} {r['mean_ms']:7.1f} ms")

    print("\n[5] confusion matrix")
    signs, conf = confusion
    pct = conf / conf.sum(axis=1, keepdims=True) * 100
    with open(RESULTS_DIR / "confusion_isl_k3.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true \\ predicted (% of test clips)"] + signs)
        for s, row in zip(signs, pct):
            w.writerow([s] + [f"{v:.1f}" for v in row])

    n_clips = {"ISL": sum(len(v) for v in isl_w.values()), "WLASL": sum(len(v) for v in wl_w.values())}
    make_plots(summary, thr, confusion, speed)
    write_markdown(summary, thr, confusion, speed, args, n_clips)
    print(f"\nwrote CSVs, plots and results.md to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
