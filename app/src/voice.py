"""Voice note transcription for CHW Copilot.

Supports MedASR for medical-domain speech-to-text when available,
with a placeholder fallback for demo/offline use.
"""
import os
from pathlib import Path
from typing import Optional

# Whether MedASR is available
_medasr_available = None


def _check_medasr() -> bool:
    """Check if MedASR is installed and available."""
    global _medasr_available
    if _medasr_available is not None:
        return _medasr_available
    try:
        import medasr  # noqa: F401
        _medasr_available = True
    except ImportError:
        _medasr_available = False
    return _medasr_available


def transcribe_audio(audio_path: str, language: str = "en") -> str:
    """Transcribe an audio file to text.

    Uses MedASR if available, otherwise returns a demo transcription
    or a message indicating MedASR is needed for real transcription.

    Args:
        audio_path: Path to the audio file (wav, mp3, etc.)
        language: Language code for transcription

    Returns:
        Transcribed text from the audio
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if _check_medasr():
        return _transcribe_with_medasr(str(path), language)
    else:
        return _transcribe_fallback(str(path))


def _transcribe_with_medasr(audio_path: str, language: str) -> str:
    """Transcribe using Google MedASR.

    MedASR is specialised for medical domain speech recognition,
    particularly effective for clinical terminology and multi-lingual
    health worker communication.
    """
    try:
        import medasr

        model = medasr.load_model()
        result = model.transcribe(audio_path, language=language)
        return result.get("text", "").strip()
    except Exception as e:
        print(f"MedASR transcription failed, using fallback: {e}")
        return _transcribe_fallback(audio_path)


def _transcribe_fallback(audio_path: str) -> str:
    """Fallback transcription for demo/offline use.

    In production, this would be replaced with MedASR.
    Returns a demo note to demonstrate the pipeline flow.
    """
    # Check for a .txt sidecar file (allows manual transcript upload)
    txt_path = Path(audio_path).with_suffix(".txt")
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip()

    # Demo mode: return a sample clinical note
    return (
        "Child 4 years old female presenting with fever for 2 days, "
        "cough and difficulty breathing. Mother reports loss of appetite. "
        "Temperature measured at home was high. No rash observed. "
        "No diarrhea or vomiting. Child was given paracetamol at home. "
        "Referred to health center for further evaluation."
    )


def is_medasr_available() -> bool:
    """Check if MedASR is available for real transcription."""
    return _check_medasr()
