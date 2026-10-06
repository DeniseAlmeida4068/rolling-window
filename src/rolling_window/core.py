from __future__ import annotations

import math
from collections import deque


class RollingWindow:
    """A bounded rolling window over a stream of numbers.

    The window holds the last `capacity` values that were pushed to it.
    Once full, each new push evicts the oldest value, so the window slides
    forward. The mean, peak (maximum) and trough (minimum) are computed
    exactly from the current contents each time they are queried.

    The implementation is deliberately direct — it keeps values in a deque and
    does a linear scan for peak/trough — because that is the simplest contract
    that cannot drift from the truth as values enter and leave. A constant-time
    monotonic-deque for min/max was rejected: it is faster, but the extra
    machinery is only worth it once the window size is large enough that a scan
    shows up in profiling. For the bounded, modest windows this library is
    aimed at, the scan is faster to reason about than to maintain.

    Edge cases:
      * A new value that equals the current peak/trough does not displace it.
        Peak and trough are tracked against the current contents, so ties are
        reported as the value itself.
      * Non-finite inputs (NaN, +inf, -inf) are accepted and participate in
        comparisons using IEEE 754 semantics. Pushing NaN makes peak() and
        trough() NaN until it leaves the window; pushing +inf sets peak() to
        +inf until it leaves. This keeps the window honest about what was fed
        to it rather than silently inventing a clamp.
      * The window is never implicitly emptied. Call `clear()` to reset.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._capacity = capacity
        self._values: deque[float] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._values)

    def push(self, value: float) -> None:
        """Append a value, evicting the oldest if the window is full."""
        self._values.append(value)

    def mean(self) -> float:
        """Arithmetic mean of the current window.

        Returns 0.0 when the window is empty, so callers don't have to guard
        against an empty window before doing arithmetic. This is the only
        place the library invents a value rather than reflecting the input;
        the alternative (raising) forces every caller to branch, which the
        stream-processing use case does not want.
        """
        n = len(self._values)
        if n == 0:
            return 0.0
        if any(math.isnan(v) for v in self._values):
            return math.nan
        try:
            return math.fsum(self._values) / n
        except ValueError:
            return math.nan

    def peak(self) -> float:
        """Maximum value currently in the window, or NaN if empty."""
        if not self._values:
            return math.nan
        if any(math.isnan(v) for v in self._values):
            return math.nan
        return max(self._values)

    def trough(self) -> float:
        """Minimum value currently in the window, or NaN if empty."""
        if not self._values:
            return math.nan
        if any(math.isnan(v) for v in self._values):
            return math.nan
        return min(self._values)

    def clear(self) -> None:
        """Remove all values from the window."""
        self._values.clear()
