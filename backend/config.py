import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SEQ_DIR = DATA_DIR / "sequences"      # raw recorded landmark sequences
MODEL_DIR = DATA_DIR / "models"       # trained keras model + label list
SIGN_DIR = DATA_DIR / "signs"         # reference clips for text -> sign

MODEL_PATH = MODEL_DIR / "sign_lstm.keras"
LABELS_PATH = MODEL_DIR / "labels.json"

# Pretrained 250-sign model: 1st place of Google's Kaggle "ASL Signs" (ISLR)
# competition, from huggingface.co/sign/kaggle-asl-signs-1st-place (MIT).
# Trained on ~94k takes by 21 signers, so it generalises to new signers far
# better than anything WLASL's ~10 clips per word can give.
ISLR_DIR = MODEL_DIR / "islr"
ISLR_MODEL_PATH = ISLR_DIR / "model.tflite"
ISLR_LABELS_PATH = ISLR_DIR / "sign_to_prediction_index_map.json"
ISLR_FPS = 30              # its training videos were ~30 fps; live frames are resampled to match
ISLR_WINDOW_SECONDS = 1.5  # median ISLR take is ~1.3 s

# "islr" (pretrained, default when downloaded) or "lstm" (trained here by ml.train)
SIGN_MODEL = os.environ.get("SIGN_MODEL", "islr" if ISLR_MODEL_PATH.exists() else "lstm")

SEQUENCE_LENGTH = 30      # frames per sample
WINDOW_SECONDS = 2.0      # live time span resampled to SEQUENCE_LENGTH (median WLASL clip ~2.1 s)
FEATURE_SIZE = 258        # pose(132) + left hand(63) + right hand(63)
SEQUENCES_PER_SIGN = 40   # how many samples to record per word
CONFIDENCE_THRESHOLD = 0.75

# ---- few-shot "Teach a new sign" (ml/fewshot.py) ----
FEWSHOT_PROTOTYPES_PATH = Path(os.environ.get("FEWSHOT_PROTOTYPES", DATA_DIR / "prototypes.json"))  # taught signs live here
# Minimum cosine similarity between a live window and a taught sign's prototype
# for it to count as a match. THE ONE PLACE TO TUNE IT. From ml/check_embeddings.py
# (3 taught clips): 0.30 accepts 69% of taught-sign clips but lets 16% of built-in
# sign clips through; 0.35 accepts 61% / 11%; 0.40 accepts 53% / 6%. Raise it if
# taught signs fire by mistake, lower it if they are missed.
# Raised from 0.35 to 0.45 after ml/evaluate.py (results/results.md, K=3): on ISL signs it keeps
# 98.5% of correct matches while wrongly accepting only ~3-7% of never-taught signs (17.7% at 0.35).
# On WLASL signs 0.45 accepts fewer correct matches (60% vs 72%).
FEWSHOT_THRESHOLD = 0.45
# Built-in vs taught (ml/combine.py). The two scores are on different scales (a
# softmax probability vs a cosine similarity), so they are not compared directly.
# Rule: a taught sign wins only if the built-in model is NOT confident, i.e. its
# top probability is below BUILTIN_CONFIDENT. Raise this to let taught signs
# override the built-in model more often; lower it to trust the built-in model more.
BUILTIN_CONFIDENT = 0.5
# Smoothing (only while signs are taught): show a sign only if it won in at least
# SMOOTH_MIN_WINS of the last SMOOTH_WINDOWS windows. Stops single noisy windows
# from flashing a wrong word.
SMOOTH_WINDOWS = 5
SMOOTH_MIN_WINS = 3
FEWSHOT_MIN_CLIPS = 3     # clips needed to teach a sign
FEWSHOT_MAX_CLIPS = 8
# Every clip recorded for a taught sign is also kept here (one .npy landmark window each)
# so accuracy can be measured later. At most EVAL_MAX_CLIPS per sign; the oldest go first.
FEWSHOT_EVAL_DIR = Path(os.environ.get("FEWSHOT_EVAL_CLIPS", DATA_DIR / "eval_clips"))
FEWSHOT_EVAL_MAX_CLIPS = 8
# A SECOND recording session of the same signs, saved by the Teach tab's "test clips" mode.
# These are never used for teaching (no prototype is written); they are for cross-session tests.
FEWSHOT_EVAL2_DIR = Path(os.environ.get("FEWSHOT_EVAL_CLIPS_SESSION2", DATA_DIR / "eval_clips_session2"))
FEWSHOT_WINDOW_FRAMES = round(ISLR_FPS * ISLR_WINDOW_SECONDS)   # 45: what the ISLR model is fed live

for d in (SEQ_DIR, MODEL_DIR, SIGN_DIR):
    d.mkdir(parents=True, exist_ok=True)
