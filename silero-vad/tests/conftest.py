"""Shared fixtures for the optional audio backends.

torchaudio and torchcodec are optional extras, so several tests need to run as
if one or both were absent. Setting a module to None in sys.modules makes a
later `import` of it raise ImportError, which is exactly what the code under
test has to cope with.
"""
import os
import sys

import pytest

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
WAV = os.path.join(DATA_DIR, "test.wav")
OPUS = os.path.join(DATA_DIR, "test.opus")
MP3 = os.path.join(DATA_DIR, "test.mp3")
# test.wav is mono 16 kHz, so reading it involves no down-mix and no resampling;
# the other two need both, which is where two backends can drift apart.
AUDIO_PATHS = [WAV, OPUS, MP3]

_TORCHCODEC_MODULES = ("torchcodec", "torchcodec.decoders", "torchcodec.encoders")


def _torchcodec_works():
    """Whether torchcodec can actually decode audio here.

    A plain import check is not enough in CI: torchcodec is pinned to a specific
    torch release but does not declare it, and it needs FFmpeg (4-9) on the
    system. Either mismatch surfaces as an arbitrary exception at import or at
    decode time, so probe the real thing and treat any failure as "unavailable".
    """
    try:
        from torchcodec.decoders import AudioDecoder

        AudioDecoder(WAV).get_all_samples()
    except Exception:
        return False
    return True


requires_torchcodec = pytest.mark.skipif(
    not _torchcodec_works(),
    reason="torchcodec is not installed, or cannot decode audio here "
           "(incompatible torch, or FFmpeg missing)",
)


@pytest.fixture
def without_torchaudio(monkeypatch):
    """Make `import torchaudio` fail, as if it were not installed."""
    monkeypatch.setitem(sys.modules, "torchaudio", None)


@pytest.fixture
def without_torchcodec(monkeypatch):
    """Make `import torchcodec` fail, as if it were not installed."""
    for name in _TORCHCODEC_MODULES:
        monkeypatch.setitem(sys.modules, name, None)
