"""Tests for mcframework.profiling module."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcframework.profiling import TorchProfilerConfig


class TestTorchProfilerConfig:
    """Unit tests for TorchProfilerConfig dataclass-like configuration."""

    def test_default_values(self):
        config = TorchProfilerConfig()
        assert config.activities == ["cpu"]
        assert config.schedule_wait == 0
        assert config.schedule_warmup == 1
        assert config.schedule_active == 3
        assert config.schedule_repeat == 2
        assert config.record_shapes is True
        assert config.profile_memory is True
        assert config.with_stack is False
        assert config.with_flops is True
        assert config.export_chrome_trace is True
        assert config.export_stacks is False
        assert config.acc_events is True
        assert config.enable_mps_profiler is False
        assert isinstance(config.output_dir, Path)
        assert str(config.output_dir) == "profiler_results"

    def test_custom_values(self):
        config = TorchProfilerConfig(
            activities=["cpu", "cuda"],
            schedule_active=10,
            output_dir="/tmp/custom_dir",
            with_stack=True,
        )
        assert config.activities == ["cpu", "cuda"]
        assert config.schedule_active == 10
        assert config.output_dir == Path("/tmp/custom_dir")
        assert config.with_stack is True

    def test_output_dir_is_path(self):
        config = TorchProfilerConfig(output_dir="some/nested/path")
        assert isinstance(config.output_dir, Path)


torch = pytest.importorskip("torch")


class TestProfiledTorchBackend:
    """Integration tests for ProfiledTorchBackend (requires PyTorch)."""

    def test_import_and_init(self, tmp_path):
        from mcframework.backends import TorchCPUBackend
        from mcframework.profiling import ProfiledTorchBackend

        base = TorchCPUBackend()
        config = TorchProfilerConfig(output_dir=str(tmp_path))
        backend = ProfiledTorchBackend(base, config)
        assert backend._backend is base

    def test_calculate_chunk_size(self, tmp_path):
        from mcframework.backends import TorchCPUBackend
        from mcframework.profiling import ProfiledTorchBackend

        base = TorchCPUBackend()
        config = TorchProfilerConfig(
            schedule_wait=1,
            schedule_warmup=1,
            schedule_active=3,
            schedule_repeat=2,
            output_dir=str(tmp_path),
        )
        backend = ProfiledTorchBackend(base, config)
        chunk = backend._calculate_chunk_size(1_000_000)
        assert chunk >= 10_000
