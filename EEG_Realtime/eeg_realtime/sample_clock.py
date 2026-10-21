"""Map a nominal 250-Hz sample counter to the host receipt clock.

The board has no hardware timestamps. A sliding lower envelope of packet
receipt offsets limits transport jitter and tracks oscillator drift without
allowing estimated samples to occur after receipt. This is a software estimate,
not a hardware synchronization guarantee. It never resamples EEG values.
"""
from collections import deque
import math

import numpy as np


CLOCK_METHOD = "receipt-envelope-5s-bounded-interpolation-v1"


class ReceiptSampleClock:
    def __init__(self, sample_rate=250, history_s=5.0):
        self.sample_rate = sample_rate
        self.history_s = history_s
        self.reset()

    def reset(self):
        self.offsets = deque()
        self.last_time = None
        self.last_receipt = None
        self.last_index = None

    def map(self, indices, received):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices) or np.any(np.diff(indices) != 1):
            raise ValueError("Clock requires consecutive nonempty sample indices")
        if not math.isfinite(received) or received < 0:
            raise ValueError("Clock requires a finite monotonic receipt time")
        if self.last_receipt is not None and received < self.last_receipt:
            raise ValueError("Packet receipt times must be monotonic")
        nominal_end = float(indices[-1]) / self.sample_rate
        self.offsets.append((received, received - nominal_end))
        while len(self.offsets) > 1 and self.offsets[0][0] < received - self.history_s:
            self.offsets.popleft()
        target_end = min(received, nominal_end + min(offset for _, offset in self.offsets))
        discontinuity = False
        if self.last_time is None:
            times = target_end - np.arange(len(indices) - 1, -1, -1) / self.sample_rate
        else:
            spacing = (target_end - self.last_time) / len(indices)
            # Normal drift adjustments stay below the decoder's 25% timestamp
            # gap limit. Bursts/large jumps are explicit discontinuities instead
            # of reversing time or silently shifting an ongoing trial's labels.
            if (indices[0] != self.last_index + 1 or
                    not .8 / self.sample_rate <= spacing <= 1.2 / self.sample_rate):
                discontinuity = True
                self.reset()
                self.offsets.append((received, received - nominal_end))
                times = received - np.arange(len(indices) - 1, -1, -1) / self.sample_rate
            else:
                times = self.last_time + np.arange(1, len(indices) + 1) * spacing
        self.last_time = float(times[-1])
        self.last_receipt = received
        self.last_index = int(indices[-1])
        return times, discontinuity
