"""When did a gauge turn? A causal 'zig-zag': a high (or low) is CONFIRMED once the gauge has moved THRESHOLD away from it, and is then
placed on the true extreme. Descriptive only, nothing here feeds a score or predicts anything. Each confirmation uses data up to that
month only (no look-ahead), but the marker sits on the earlier extreme, so a turn only shows up on a chart once it is confirmed.

THRESHOLD = 0.4 on the gauge's -2..+2 scale (a tenth of its range). Chosen from how often it fires and how quickly, not by fitting to market
returns. Measured on 1995-2026 (recomputed for the 8 gauges and the headline): a turn every 4.5 months (investor mood) to about 20 months
(distressed debt), typically 7-15; confirmed a median of 2-3 months after the extreme (3-6 for the quarterly, slow gauges; the slowest took
18). On noisy gauges half the turns follow the previous one within 3 months, i.e. they often reverse.
"""
THRESHOLD = 0.4
EPS = 1e-9


def find_turns(values: list, threshold: float = THRESHOLD) -> dict:
    """values: chronological gauge scores (no NaN). Returns
        turns  : [(kind, extreme_index, confirmed_index)], kind 'high' or 'low', in time order
        trend  : 'up' (last confirmed turn was a low), 'down' (a high) or None (no turn confirmed yet)
        extreme: index of the running extreme since the last confirmed turn (the high while rising, the low while falling); None if no trend yet"""
    n, turns, trend, hi, lo = len(values), [], 0, 0, 0
    for i in range(1, n):
        v = values[i]
        if trend == 0:                                   # nothing confirmed yet: watch both ends
            hi, lo = (i if v > values[hi] else hi), (i if v < values[lo] else lo)
            if v - values[lo] >= threshold - EPS:
                turns.append(("low", lo, i)); trend, hi = 1, i
            elif values[hi] - v >= threshold - EPS:
                turns.append(("high", hi, i)); trend, lo = -1, i
        elif trend == 1:                                 # rising since a confirmed low: track the high
            if v > values[hi]:
                hi = i
            elif values[hi] - v >= threshold - EPS:
                turns.append(("high", hi, i)); trend, lo = -1, i
        else:                                            # falling since a confirmed high: track the low
            if v < values[lo]:
                lo = i
            elif v - values[lo] >= threshold - EPS:
                turns.append(("low", lo, i)); trend, hi = 1, i
    return dict(turns=turns, trend={1: "up", -1: "down", 0: None}[trend], extreme={1: hi, -1: lo, 0: None}[trend])
