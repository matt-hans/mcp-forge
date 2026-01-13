"""Unit tests for LatencyStats class."""

import pytest

from mcp_forge.validation.benchmark import LatencyStats


class TestLatencyStats:
    """Tests for LatencyStats dataclass."""

    def test_latency_stats_empty_list(self):
        """LatencyStats handles empty list gracefully."""
        stats = LatencyStats.from_samples([])
        assert stats.min_ms == 0.0
        assert stats.max_ms == 0.0
        assert stats.mean_ms == 0.0
        assert stats.p50_ms == 0.0
        assert stats.p95_ms == 0.0
        assert stats.p99_ms == 0.0

    def test_latency_stats_single_sample(self):
        """LatencyStats handles single sample."""
        stats = LatencyStats.from_samples([100.0])
        assert stats.min_ms == 100.0
        assert stats.max_ms == 100.0
        assert stats.mean_ms == 100.0
        assert stats.p50_ms == 100.0
        # P95/P99 fall back to max for small samples
        assert stats.p95_ms == 100.0
        assert stats.p99_ms == 100.0

    def test_latency_stats_two_samples(self):
        """LatencyStats handles two samples."""
        stats = LatencyStats.from_samples([50.0, 150.0])
        assert stats.min_ms == 50.0
        assert stats.max_ms == 150.0
        assert stats.mean_ms == 100.0
        # P50 is index 1 (150.0) for sorted list [50, 150]
        assert stats.p50_ms == 150.0

    def test_latency_stats_percentiles_basic(self):
        """LatencyStats calculates basic percentiles."""
        # 10 samples: 10, 20, 30, ..., 100
        latencies = [float(i * 10) for i in range(1, 11)]
        stats = LatencyStats.from_samples(latencies)

        assert stats.min_ms == 10.0
        assert stats.max_ms == 100.0
        assert stats.mean_ms == 55.0  # Average of 10+20+...+100
        assert stats.p50_ms == 60.0  # Index 5 of 10 items

    def test_latency_stats_percentiles_large_sample(self):
        """LatencyStats calculates correct percentiles for large samples."""
        # 100 samples: 1, 2, 3, ..., 100
        latencies = [float(i) for i in range(1, 101)]
        stats = LatencyStats.from_samples(latencies)

        assert stats.min_ms == 1.0
        assert stats.max_ms == 100.0
        assert stats.mean_ms == 50.5  # Average of 1-100
        assert stats.p50_ms == 51.0  # Index 50
        assert stats.p95_ms == 96.0  # Index 95
        assert stats.p99_ms == 100.0  # Index 99

    def test_latency_stats_p95_threshold(self):
        """P95 uses max for samples < 20."""
        # 19 samples
        latencies = [float(i) for i in range(1, 20)]
        stats = LatencyStats.from_samples(latencies)
        assert stats.p95_ms == 19.0  # Falls back to max

        # 20 samples: values 1.0 to 20.0
        # int(20 * 0.95) = 19, so index 19 = value 20.0
        latencies = [float(i) for i in range(1, 21)]
        stats = LatencyStats.from_samples(latencies)
        assert stats.p95_ms == 20.0  # Index 19 (0-based) = 20th value

    def test_latency_stats_p99_threshold(self):
        """P99 uses max for samples < 100."""
        # 99 samples
        latencies = [float(i) for i in range(1, 100)]
        stats = LatencyStats.from_samples(latencies)
        assert stats.p99_ms == 99.0  # Falls back to max

        # 100 samples
        latencies = [float(i) for i in range(1, 101)]
        stats = LatencyStats.from_samples(latencies)
        assert stats.p99_ms == 100.0  # Index int(100 * 0.99) = 99

    def test_latency_stats_to_dict(self):
        """LatencyStats serializes correctly."""
        stats = LatencyStats(
            min_ms=10.0,
            max_ms=100.0,
            mean_ms=55.0,
            p50_ms=50.0,
            p95_ms=95.0,
            p99_ms=99.0,
        )
        data = stats.to_dict()

        assert data == {
            "min_ms": 10.0,
            "max_ms": 100.0,
            "mean_ms": 55.0,
            "p50_ms": 50.0,
            "p95_ms": 95.0,
            "p99_ms": 99.0,
        }

    def test_latency_stats_unsorted_input(self):
        """LatencyStats sorts input correctly."""
        # Unsorted input
        latencies = [50.0, 10.0, 90.0, 30.0, 70.0]
        stats = LatencyStats.from_samples(latencies)

        # Should be sorted: [10, 30, 50, 70, 90]
        assert stats.min_ms == 10.0
        assert stats.max_ms == 90.0
        assert stats.mean_ms == 50.0
        assert stats.p50_ms == 50.0  # Index 2

    def test_latency_stats_identical_values(self):
        """LatencyStats handles identical values."""
        latencies = [100.0] * 10
        stats = LatencyStats.from_samples(latencies)

        assert stats.min_ms == 100.0
        assert stats.max_ms == 100.0
        assert stats.mean_ms == 100.0
        assert stats.p50_ms == 100.0
        assert stats.p95_ms == 100.0
        assert stats.p99_ms == 100.0

    def test_latency_stats_floats_precision(self):
        """LatencyStats maintains float precision."""
        latencies = [0.5, 1.5, 2.5]
        stats = LatencyStats.from_samples(latencies)

        assert stats.min_ms == 0.5
        assert stats.max_ms == 2.5
        assert stats.mean_ms == 1.5
