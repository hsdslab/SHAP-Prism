"""Validated, atomic writes for publication figure files."""

from __future__ import annotations

import errno
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

from PIL import Image
from pypdf import PdfReader


class ArtifactValidationError(ValueError):
    """Raised when an output file cannot be decoded as its declared format."""


def _temporary_parent(target_parent: Path) -> Path:
    """Keep transient files outside the output tree when atomic rename permits."""

    configured = os.environ.get("SHAP_PRISM_TEMP_DIR")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    system_temp = Path(tempfile.gettempdir())
    try:
        if system_temp.stat().st_dev == target_parent.stat().st_dev:
            return system_temp
    except OSError:
        pass
    return target_parent


def _publish(temporary: Path, target: Path) -> None:
    """Publish by rename, with a target-filesystem staging fallback."""

    try:
        os.replace(temporary, target)
        return
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise

    descriptor, local_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=target.suffix, dir=target.parent
    )
    os.close(descriptor)
    local = Path(local_name)
    try:
        shutil.copyfile(temporary, local)
        validate_figure_file(local)
        os.replace(local, target)
    finally:
        local.unlink(missing_ok=True)


def validate_png(path: str | Path) -> Path:
    """Decode a PNG and require a nonempty raster."""

    candidate = Path(path)
    if not candidate.is_file() or candidate.stat().st_size == 0:
        raise ArtifactValidationError(f"PNG is missing or empty: {candidate}")
    try:
        with Image.open(candidate) as image:
            if image.format != "PNG":
                raise ArtifactValidationError(f"File is not a PNG: {candidate}")
            image.verify()
        with Image.open(candidate) as image:
            image.load()
            width, height = image.size
    except ArtifactValidationError:
        raise
    except Exception as error:
        raise ArtifactValidationError(f"PNG cannot be decoded: {candidate}") from error
    if width < 1 or height < 1:
        raise ArtifactValidationError(f"PNG has invalid dimensions: {candidate}")
    return candidate


def validate_pdf(path: str | Path) -> Path:
    """Parse a PDF and require at least one page with a valid media box."""

    candidate = Path(path)
    if not candidate.is_file() or candidate.stat().st_size == 0:
        raise ArtifactValidationError(f"PDF is missing or empty: {candidate}")
    try:
        reader = PdfReader(str(candidate), strict=True)
        if len(reader.pages) < 1:
            raise ArtifactValidationError(f"PDF contains no pages: {candidate}")
        for page in reader.pages:
            box = page.mediabox
            if float(box.width) <= 0 or float(box.height) <= 0:
                raise ArtifactValidationError(f"PDF has an invalid media box: {candidate}")
    except ArtifactValidationError:
        raise
    except Exception as error:
        raise ArtifactValidationError(f"PDF cannot be parsed: {candidate}") from error
    return candidate


def validate_figure_file(path: str | Path) -> Path:
    """Validate one supported publication figure file."""

    candidate = Path(path)
    suffix = candidate.suffix.casefold()
    if suffix == ".png":
        return validate_png(candidate)
    if suffix == ".pdf":
        return validate_pdf(candidate)
    raise ArtifactValidationError(f"Unsupported figure format: {candidate}")


def atomic_save_figure(
    figure,
    targets: Iterable[str | Path],
    *,
    png_dpi: int,
    pdf_dpi: int = 600,
) -> list[Path]:
    """Write, decode, and atomically publish every requested figure file.

    Existing target files remain untouched unless all temporary outputs pass
    structural format validation.
    """

    resolved = [Path(target) for target in targets]
    if not resolved:
        return []
    if len(set(resolved)) != len(resolved):
        raise ValueError("figure output targets must be unique")
    for target in resolved:
        if target.suffix.casefold() not in {".png", ".pdf"}:
            raise ValueError("figure output suffix must be .png or .pdf")
        target.parent.mkdir(parents=True, exist_ok=True)

    temporary_paths: list[tuple[Path, Path]] = []
    try:
        for target in resolved:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f"shap-prism-{target.stem}.",
                suffix=target.suffix,
                dir=_temporary_parent(target.parent),
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            temporary_paths.append((target, temporary))
            figure.savefig(
                temporary,
                dpi=png_dpi if target.suffix.casefold() == ".png" else pdf_dpi,
                facecolor="white",
            )
            validate_figure_file(temporary)
        for target, temporary in temporary_paths:
            _publish(temporary, target)
    finally:
        for _target, temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    return resolved
