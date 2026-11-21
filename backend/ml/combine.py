"""Merge the built-in 250-sign recognition with signs the user has taught.

Per live window we have two opinions:
  - built-in model: a label with a softmax probability (0-1)
  - taught signs:   the closest prototype with a cosine similarity, or None if
                    nothing is close enough (ml/fewshot.py applies FEWSHOT_THRESHOLD)
Those two numbers are on different scales, so they are not compared with each
other. A simple rule decides instead (see decide), and a smoother then stops
single noisy windows from flashing a wrong word.
"""
from collections import Counter, deque

from config import BUILTIN_CONFIDENT, CONFIDENCE_THRESHOLD, SMOOTH_MIN_WINS, SMOOTH_WINDOWS

NOT_SURE = "Not sure"

# (source, label, confidence); source is "built-in", "taught", or None for "Not sure"
Result = tuple[str | None, str, float]


def decide(builtin_label: str, builtin_conf: float, taught: tuple[str, float] | None) -> Result:
    """One window -> one result.

    1. A taught sign wins if it matched (similarity passed FEWSHOT_THRESHOLD) AND the
       built-in model is not confident (top probability < BUILTIN_CONFIDENT). If the
       built-in model is fairly sure, we trust it: it knows 250 signs from 21 signers.
    2. Otherwise the built-in label is used if it clears the usual
       CONFIDENCE_THRESHOLD.
    3. Otherwise: "Not sure".
    (A built-in probability between BUILTIN_CONFIDENT and CONFIDENCE_THRESHOLD is
    too high to hand over to a taught sign but too low to show: also "Not sure".)
    """
    if taught is not None and builtin_conf < BUILTIN_CONFIDENT:
        name, similarity = taught
        return "taught", name, similarity
    if builtin_conf >= CONFIDENCE_THRESHOLD:
        return "built-in", builtin_label, builtin_conf
    return None, NOT_SURE, builtin_conf


class Smoother:
    """Majority vote over the last few windows (one instance per connection).

    A sign is only shown once it has won in at least `min_wins` of the last `size`
    windows; its confidence is the average over the windows it won."""

    def __init__(self, size: int = SMOOTH_WINDOWS, min_wins: int = SMOOTH_MIN_WINS):
        if not 1 <= min_wins <= size:
            raise ValueError("need 1 <= SMOOTH_MIN_WINS <= SMOOTH_WINDOWS")
        self.min_wins = min_wins
        self.history: deque[Result] = deque(maxlen=size)

    def update(self, result: Result) -> Result:
        """Add this window's result; return the smoothed result."""
        self.history.append(result)
        # count wins per (source, label), ignoring "Not sure" windows
        votes = Counter((s, l) for s, l, _ in self.history if s is not None)
        if votes:
            (source, label), wins = votes.most_common(1)[0]
            if wins >= self.min_wins:
                confs = [c for s, l, c in self.history if (s, l) == (source, label)]
                return source, label, sum(confs) / len(confs)
        return None, NOT_SURE, result[2]

    def reset(self) -> None:
        self.history.clear()
