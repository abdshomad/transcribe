"""Tests for EngineRegistry, dynamic resolution, and factory instantiation."""

import pytest
from typing import Callable, List, Optional, Tuple
from transcribe.engines.base import BaseTranscriber
from transcribe.engines.factory import (
    EngineRegistry,
    _resolve_engine_class,
    get_transcriber,
)
from transcribe.models import TranscriptSegment


class MockEngine(BaseTranscriber):
    """Mock engine for registry testing."""

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        on_segment: Optional[Callable[[TranscriptSegment], None]] = None,
        **kwargs,
    ) -> Tuple[List[TranscriptSegment], str, float]:
        return [], language or "en", 1.0


class AnotherMockEngine(BaseTranscriber):
    """Another mock engine for testing class resolution."""

    def transcribe(self, *args, **kwargs):
        return [], "id", 0.95


def test_registry_register_and_instantiate_class():
    """Test registering a direct class and instantiating."""
    reg = EngineRegistry()
    reg.register("mock-model", MockEngine)
    engine = reg.get_transcriber("mock-model", device="cpu")
    assert isinstance(engine, MockEngine)
    assert engine.model_name == "mock-model"
    assert engine.device == "cpu"


def test_registry_heuristic_resolution():
    """Test heuristic model name resolution."""
    reg = EngineRegistry()
    reg.register("custom-wav2vec", MockEngine)
    assert reg.get_engine_class("custom-wav2vec") == MockEngine

    # Direct registration for testing dotpath resolution
    reg.register("custom-dotpath", "transcribe.engines.base.BaseTranscriber")
    # BaseTranscriber is an abstract class subclass of BaseTranscriber
    cls = reg.get_engine_class("custom-dotpath")
    assert cls == BaseTranscriber


def test_registry_list_supported_models():
    """Test list_supported_models returns registered keys."""
    reg = EngineRegistry()
    models = reg.list_supported_models()
    assert "tiny" in models
    assert "indonesian-wav2vec2-regional" in models
    assert "sensevoice-small" in models
    assert "moonshine-base" in models


def test_resolve_engine_class_errors():
    """Test invalid target error handling in _resolve_engine_class."""
    with pytest.raises(ValueError):
        _resolve_engine_class(12345)  # type: ignore

    with pytest.raises(TypeError):
        # A class that is not a subclass of BaseTranscriber
        _resolve_engine_class("transcribe.models.WordInfo")


def test_global_get_transcriber_with_custom_registration():
    """Test module-level get_transcriber convenience function."""
    from transcribe.engines.factory import default_registry

    default_registry.register("test-global-mock", MockEngine)
    instance = get_transcriber("test-global-mock", device="cpu")
    assert isinstance(instance, MockEngine)
    assert instance.model_name == "test-global-mock"
