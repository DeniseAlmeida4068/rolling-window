import math
import unittest

from rolling_window import RollingWindow


class TestRollingWindow(unittest.TestCase):
    def test_empty_window(self):
        w = RollingWindow(4)
        self.assertEqual(len(w), 0)
        self.assertEqual(w.capacity, 4)
        self.assertEqual(w.mean(), 0.0)
        self.assertTrue(math.isnan(w.peak()))
        self.assertTrue(math.isnan(w.trough()))

    def test_below_capacity_keeps_all(self):
        w = RollingWindow(5)
        w.push(1.0)
        w.push(2.0)
        w.push(3.0)
        self.assertEqual(len(w), 3)
        self.assertAlmostEqual(w.mean(), 2.0)
        self.assertEqual(w.peak(), 3.0)
        self.assertEqual(w.trough(), 1.0)

    def test_full_window_then_evicts_oldest(self):
        w = RollingWindow(3)
        for v in (1.0, 2.0, 3.0, 4.0):
            w.push(v)
        self.assertEqual(len(w), 3)
        # Window now holds 2.0, 3.0, 4.0.
        self.assertAlmostEqual(w.mean(), 3.0)
        self.assertEqual(w.peak(), 4.0)
        self.assertEqual(w.trough(), 2.0)

    def test_peak_and_trough_track_when_extreme_evicts(self):
        w = RollingWindow(3)
        w.push(10.0)
        w.push(-5.0)
        w.push(7.0)
        self.assertEqual(w.peak(), 10.0)
        self.assertEqual(w.trough(), -5.0)
        w.push(0.0)  # evicts 10.0
        self.assertEqual(w.peak(), 7.0)
        self.assertEqual(w.trough(), -5.0)
        w.push(0.0)  # evicts -5.0
        self.assertEqual(w.trough(), 0.0)
        self.assertEqual(w.peak(), 7.0)

    def test_negative_numbers(self):
        w = RollingWindow(4)
        w.push(-1.0)
        w.push(-10.0)
        w.push(-3.0)
        self.assertEqual(w.peak(), -1.0)
        self.assertEqual(w.trough(), -10.0)
        self.assertAlmostEqual(w.mean(), -14.0 / 3.0)

    def test_ties_report_value_itself(self):
        w = RollingWindow(3)
        w.push(5.0)
        w.push(5.0)
        w.push(5.0)
        self.assertEqual(w.peak(), 5.0)
        self.assertEqual(w.trough(), 5.0)
        self.assertAlmostEqual(w.mean(), 5.0)

    def test_repeated_pushes(self):
        w = RollingWindow(2)
        for _ in range(100):
            w.push(7.0)
        self.assertEqual(len(w), 2)
        self.assertAlmostEqual(w.mean(), 7.0)
        self.assertEqual(w.peak(), 7.0)
        self.assertEqual(w.trough(), 7.0)

    def test_nan_in_window(self):
        w = RollingWindow(3)
        w.push(1.0)
        w.push(math.nan)
        w.push(3.0)
        # NaN is contagious in max/min under Python's default ordering;
        # peak and trough reflect the stream honestly, so they become NaN.
        self.assertTrue(math.isnan(w.peak()))
        self.assertTrue(math.isnan(w.trough()))
        self.assertTrue(math.isnan(w.mean()))

    def test_inf_pushed(self):
        w = RollingWindow(3)
        w.push(1.0)
        w.push(math.inf)
        w.push(-math.inf)
        self.assertEqual(w.peak(), math.inf)
        self.assertEqual(w.trough(), -math.inf)
        # +inf + (-inf) is NaN under IEEE 754 semantics.
        self.assertTrue(math.isnan(w.mean()))

    def test_capacity_one(self):
        w = RollingWindow(1)
        w.push(42.0)
        self.assertEqual(len(w), 1)
        self.assertEqual(w.peak(), 42.0)
        self.assertEqual(w.trough(), 42.0)
        self.assertAlmostEqual(w.mean(), 42.0)
        w.push(-1.0)
        self.assertEqual(w.peak(), -1.0)
        self.assertEqual(w.trough(), -1.0)
        self.assertAlmostEqual(w.mean(), -1.0)

    def test_clear(self):
        w = RollingWindow(3)
        w.push(1.0)
        w.push(2.0)
        w.clear()
        self.assertEqual(len(w), 0)
        self.assertEqual(w.mean(), 0.0)
        self.assertTrue(math.isnan(w.peak()))
        self.assertTrue(math.isnan(w.trough()))
        # Window still usable after clear.
        w.push(9.0)
        self.assertEqual(w.peak(), 9.0)

    def test_rejects_non_positive_capacity(self):
        with self.assertRaises(ValueError):
            RollingWindow(0)
        with self.assertRaises(ValueError):
            RollingWindow(-1)

    def test_mean_uses_fsum_for_precision(self):
        # A naive sum would lose precision here; fsum keeps the tail alive.
        w = RollingWindow(4)
        w.push(1e16)
        w.push(1.0)
        w.push(-1e16)
        w.push(1.0)
        self.assertAlmostEqual(w.mean(), 0.5)


if __name__ == "__main__":
    unittest.main()
