"""Unit tests for merging built-in and taught signs (ml/combine.py) and the
debounce they share with the built-in path. No model, webcam or server needed.

    cd backend && python -m tests.combine_test
"""
from config import BUILTIN_CONFIDENT, CONFIDENCE_THRESHOLD, SMOOTH_MIN_WINS, SMOOTH_WINDOWS
from ml.combine import NOT_SURE, Smoother, decide
from ml.predict import FrameBuffer

failures = []


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        failures.append(name)


def test_decide():
    print("decide (one window)")
    low, mid, high = BUILTIN_CONFIDENT - 0.1, (BUILTIN_CONFIDENT + CONFIDENCE_THRESHOLD) / 2, CONFIDENCE_THRESHOLD + 0.1
    taught = ("cool", 0.5)
    check("taught wins when built-in is not confident", decide("drink", low, taught) == ("taught", "cool", 0.5))
    check("built-in wins when it is confident, even if taught matched", decide("drink", high, taught) == ("built-in", "drink", high))
    check("in-between built-in confidence: not sure (neither side wins)", decide("drink", mid, taught)[:2] == (None, NOT_SURE))
    check("no taught match + confident built-in -> built-in", decide("drink", high, None) == ("built-in", "drink", high))
    check("no taught match + weak built-in -> not sure", decide("drink", low, None)[:2] == (None, NOT_SURE))


def test_smoother():
    print(f"Smoother ({SMOOTH_MIN_WINS} of last {SMOOTH_WINDOWS})")
    win = ("taught", "cool", 0.4)
    nope = (None, NOT_SURE, 0.2)

    s = Smoother()
    outs = [s.update(win) for _ in range(SMOOTH_MIN_WINS)]
    check("not shown before enough wins", all(o[1] == NOT_SURE for o in outs[:-1]))
    check("shown once it has won enough windows", outs[-1][:2] == ("taught", "cool"))

    s = Smoother()
    s.update(win), s.update(nope), s.update(win), s.update(nope)
    check("scattered wins below the minimum stay hidden", s.update(nope)[1] == NOT_SURE)

    s = Smoother()
    for r in (win, ("taught", "cool", 0.6), win):
        out = s.update(r)
    check("confidence is the average over winning windows", abs(out[2] - (0.4 + 0.6 + 0.4) / 3) < 1e-9)

    s = Smoother()
    for r in [("built-in", "drink", 0.9)] * 3:
        s.update(r)
    outs = [s.update(win) for _ in range(SMOOTH_MIN_WINS)]
    check("old sign is replaced once a new one has the votes", outs[-1][:2] == ("taught", "cool"))

    s = Smoother()
    for _ in range(SMOOTH_MIN_WINS):
        s.update(win)
    s.reset()
    check("reset forgets the history", s.update(win)[1] == NOT_SURE)

    try:
        Smoother(size=3, min_wins=4)
        check("min_wins greater than size is rejected", False)
    except ValueError:
        check("min_wins greater than size is rejected", True)


def test_debounce_shared():
    print("FrameBuffer.accept_label (debounce shared by both paths)")
    b = FrameBuffer(window_seconds=2.0)
    check("first result emits", b.accept_label("cool", t=1.0) is True)
    check("same label repeated is suppressed", b.accept_label("cool", t=1.1) is False)
    check("nothing confident emits nothing", b.accept_label(None, t=1.2) is False)
    check("one gap mid-gesture is not a new word", b.accept_label("cool", t=1.3) is False)
    b.accept_label(None, t=2.0)
    b.accept_label(None, t=4.0)
    check("same sign emits again after a full quiet window", b.accept_label("cool", t=4.1) is True)
    check("a different sign emits immediately", b.accept_label("drink", t=4.2) is True)

    # accept() must still behave as it did before the refactor
    b = FrameBuffer(window_seconds=2.0)
    check("accept: above threshold emits", b.accept("hi", CONFIDENCE_THRESHOLD + 0.1, t=1.0) is True)
    check("accept: below threshold does not", b.accept("yo", CONFIDENCE_THRESHOLD - 0.1, t=1.1) is False)


if __name__ == "__main__":
    test_decide()
    test_smoother()
    test_debounce_shared()
    print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
    raise SystemExit(1 if failures else 0)
