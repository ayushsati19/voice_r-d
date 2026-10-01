"""Tests for the optional dependencies.

torchaudio, torchcodec, onnxruntime and numpy are all optional extras: the VAD
model itself needs none of them. These tests pin down what happens when they
are missing, and make sure importing the package does not drag them in.
"""
import subprocess
import sys

import pytest
import torch

from conftest import WAV, requires_torchcodec

from silero_vad import read_audio, save_audio


def _run_python(code):
    """Run a snippet in a fresh interpreter, returning its stdout."""
    return subprocess.check_output([sys.executable, "-c", code], text=True)


def test_import_does_not_pull_optional_packages():
    """`import silero_vad` must not import the optional stack.

    The sequence path needs numpy and onnxruntime, so it is imported lazily; a
    plain `import silero_vad` has to stay usable with neither installed. Run in
    a subprocess so other tests, which do use those paths, cannot mask a
    regression here.
    """
    code = ("import silero_vad, sys; "
            "assert 'silero_vad.sequence_vad' not in sys.modules, 'sequence_vad imported eagerly'; "
            "assert 'onnxruntime' not in sys.modules, 'onnxruntime imported eagerly'; "
            "assert 'torchaudio' not in sys.modules, 'torchaudio imported eagerly'")
    _run_python(code)


def test_onnxruntime_imported_only_when_a_session_is_built():
    """Reaching the sequence module must not yet import onnxruntime."""
    code = ("import silero_vad, sys; "
            "silero_vad.get_speech_timestamps_sequence; "
            "assert 'silero_vad.sequence_vad' in sys.modules, 'lazy attr did not import the module'; "
            "assert 'onnxruntime' not in sys.modules, 'onnxruntime imported at module level'; "
            "silero_vad.load_silero_vad(sequence=True); "
            "assert 'onnxruntime' in sys.modules, 'onnxruntime never imported'")
    _run_python(code)


def test_lazy_sequence_attributes_still_resolve():
    import silero_vad

    assert "get_speech_timestamps_sequence" in dir(silero_vad)
    from silero_vad import SileroVADSequence, get_speech_timestamps_sequence

    assert SileroVADSequence is silero_vad.SileroVADSequence
    assert callable(get_speech_timestamps_sequence)

    with pytest.raises(AttributeError):
        silero_vad.definitely_not_an_attribute


def test_no_audio_backend_raises_actionable_error(without_torchaudio, without_torchcodec):
    with pytest.raises(ImportError) as excinfo:
        read_audio(WAV, sampling_rate=16000)
    message = str(excinfo.value)
    assert "silero-vad[audio]" in message
    assert "silero-vad[codec]" in message

    with pytest.raises(ImportError) as excinfo:
        save_audio("unused.wav", torch.zeros(16000), sampling_rate=16000)
    assert "silero-vad[codec]" in str(excinfo.value)


def test_vad_runs_without_any_audio_backend(without_torchaudio, without_torchcodec):
    """The model must stay usable when audio comes from somewhere else."""
    from silero_vad import load_silero_vad, get_speech_timestamps

    model = load_silero_vad()
    audio = torch.zeros(16000, dtype=torch.float32)
    assert get_speech_timestamps(audio, model, return_seconds=True) == []


@requires_torchcodec
def test_torchcodec_backend_without_torchaudio(without_torchaudio, tmp_path):
    """torchcodec alone must be enough for both reading and writing."""
    audio = read_audio(WAV, sampling_rate=16000)

    assert audio.dim() == 1
    assert audio.dtype == torch.float32
    assert audio.numel() > 0

    out = tmp_path / "out.wav"
    save_audio(str(out), audio[:16000], sampling_rate=16000)
    assert out.stat().st_size > 0
    assert read_audio(str(out), sampling_rate=16000).numel() == 16000
