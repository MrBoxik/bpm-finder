from __future__ import annotations

from pathlib import Path


SUPPORTED_AUDIO_EXTENSIONS = (
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
    ".aiff",
    ".aif",
    ".aifc",
    ".au",
    ".snd",
    ".caf",
    ".w64",
    ".rf64",
    ".voc",
    ".m4a",
    ".aac",
    ".wma",
)

AUDIO_FILE_DIALOG_PATTERN = " ".join(
    f"*{extension}" for extension in SUPPORTED_AUDIO_EXTENSIONS
)


def is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
