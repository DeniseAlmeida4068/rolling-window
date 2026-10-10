# Rolling Window

A bounded rolling window over a stream of numbers, exposing the current mean, peak and trough. Standard library only.

```python
from rolling_window import RollingWindow

w = RollingWindow(capacity=3)
w.push(1.0)
w.push(5.0)
w.push(2.0)
w.push(9.0)  # evicts 1.0
w.mean()   # 5.333...
w.peak()   # 9.0
w.trough() # 2.0
len(w)     # 3
w.clear()
```

## Why

For stream processing where you need a quick, correct summary of the last *N* values — latency pings, sensor readings, queue depths — without pulling in NumPy or maintaining a ring buffer yourself. The trade-off is simplicity over throughput: peak and trough are recomputed by a linear scan on each call rather than via a monotonic deque, so calls are O(window size). That is fine for modest windows and avoids the bookkeeping that makes a monotonic deque easy to get subtly wrong. If you need millions of calls per second on a thousand-wide window, this is not the library for it.

## Edge cases

- `mean()` on an empty window returns `0.0` rather than raising, so callers can do arithmetic without a guard. `peak()` and `trough()` on an empty window return `math.nan`.
- Non-finite inputs (`math.nan`, `math.inf`) are accepted as-is and participate in comparisons under IEEE 754 semantics. Pushing `NaN` makes `peak()`, `trough()` and `mean()` return `NaN` until it leaves the window. The library reports what was fed to it; it does not clamp or drop values silently.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

