from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from config.settings import AnalyzerSettings


LOGGER = logging.getLogger(__name__)


def discover_code_files(folder_path: str, settings: AnalyzerSettings) -> dict[str, Any]:
    root = Path(folder_path).expanduser().resolve()
    discovered_files: list[dict[str, Any]] = []
    skipped_large_files: list[str] = []

    for path in root.rglob("*"):
        if path.is_dir():
            continue

        relative_parts = path.relative_to(root).parts
        if any(part in settings.ignored_directories for part in relative_parts[:-1]):
            continue
        if settings.should_ignore_file(path.name):
            continue
        if not settings.is_allowed_extension(path):
            continue

        try:
            size = path.stat().st_size
        except OSError as exc:
            LOGGER.warning("Unable to stat %s: %s", path, exc)
            continue

        if size > settings.max_file_bytes:
            skipped_large_files.append(str(path.relative_to(root)))
            continue

        discovered_files.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(root)),
                "extension": "".join(path.suffixes) or path.suffix,
                "size_bytes": size,
            }
        )

    discovered_files.sort(key=lambda item: item["relative_path"])
    return {
        "files": discovered_files,
        "skipped_large_files": skipped_large_files,
    }
