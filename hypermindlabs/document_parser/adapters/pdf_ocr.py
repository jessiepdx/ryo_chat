from __future__ import annotations

import time
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from hypermindlabs.document_parser.adapters.base import (
    DocumentParseProfile,
    DocumentParserAdapter,
    clamp_confidence,
    clamp_cost,
)

try:
    from pypdf import PdfReader
except Exception:  # noqa: BLE001
    PdfReader = None  # type: ignore[assignment]

try:
    import pytesseract
except Exception:  # noqa: BLE001
    pytesseract = None  # type: ignore[assignment]

try:
    from PIL import Image
except Exception:  # noqa: BLE001
    Image = None  # type: ignore[assignment]


OcrProvider = Callable[..., dict[str, Any] | str | None]


def _runtime_value(config_manager: Any | None, path: str, default: Any) -> Any:
    if config_manager is None:
        return default
    try:
        return config_manager.runtimeValue(path, default)
    except Exception:  # noqa: BLE001
        return default


def _runtime_bool(config_manager: Any | None, path: str, default: bool) -> bool:
    value = _runtime_value(config_manager, path, default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _runtime_float(config_manager: Any | None, path: str, default: float) -> float:
    try:
        return float(_runtime_value(config_manager, path, default))
    except (TypeError, ValueError):
        return float(default)


def _runtime_int(config_manager: Any | None, path: str, default: int) -> int:
    try:
        return int(_runtime_value(config_manager, path, default))
    except (TypeError, ValueError):
        return int(default)


def _safe_text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _coerce_provider_result(result: dict[str, Any] | str | None) -> dict[str, Any]:
    if isinstance(result, dict):
        return dict(result)
    if isinstance(result, str):
        return {
            "text": result,
            "confidence": 0.65,
            "available": True,
            "engine": "custom-provider",
        }
    return {
        "text": "",
        "confidence": 0.0,
        "available": False,
        "engine": "unavailable",
    }


class PdfOcrParserAdapter(DocumentParserAdapter):
    adapter_name = "pdf-ocr"
    adapter_version = "v1"

    def __init__(
        self,
        *,
        config_manager: Any | None = None,
        ocr_provider: OcrProvider | None = None,
    ):
        self._config = config_manager
        self._ocr_provider = ocr_provider if callable(ocr_provider) else self._default_ocr_provider

    def can_parse(self, profile: DocumentParseProfile) -> bool:
        if not _runtime_bool(self._config, "documents.pdf_ocr_enabled", True):
            return False
        mime = str(profile.file_mime or "").strip().lower()
        probes = profile.probes if isinstance(profile.probes, dict) else {}
        if mime == "application/pdf":
            return True
        if profile.file_extension == ".pdf":
            return True
        return bool(probes.get("pdf_header"))

    def confidence(self, profile: DocumentParseProfile) -> float:
        score = 0.42
        mime = str(profile.file_mime or "").strip().lower()
        probes = profile.probes if isinstance(profile.probes, dict) else {}
        if mime == "application/pdf":
            score += 0.25
        if profile.file_extension == ".pdf":
            score += 0.12
        if probes.get("pdf_header"):
            score += 0.12
        if probes.get("sample_text"):
            score -= 0.08
        score -= max(0.0, float(profile.complexity_score or 0.0) - 0.8) * 0.15
        return clamp_confidence(score, default=0.0)

    def cost(self, profile: DocumentParseProfile) -> float:
        size_mb = max(0.0, float(profile.file_size_bytes or 0) / (1024.0 * 1024.0))
        return clamp_cost(3.4 + (size_mb / 1.8), default=3.4)

    def _default_ocr_provider(
        self,
        *,
        page: Any,
        page_number: int,
        direct_text: str,
        profile: DocumentParseProfile,
    ) -> dict[str, Any]:
        if pytesseract is None or Image is None:
            return {
                "text": "",
                "confidence": 0.0,
                "available": False,
                "engine": "unavailable",
                "warning": "ocr_engine_unavailable",
            }

        image_candidates: list[Any] = []
        try:
            image_candidates = list(getattr(page, "images", []) or [])
        except Exception:  # noqa: BLE001
            image_candidates = []

        chunks: list[str] = []
        for image_ref in image_candidates:
            pil_image = None
            try:
                pil_image = getattr(image_ref, "image", None)
                if pil_image is None:
                    raw_data = getattr(image_ref, "data", b"")
                    if raw_data:
                        pil_image = Image.open(BytesIO(raw_data))
            except Exception:  # noqa: BLE001
                pil_image = None
            if pil_image is None:
                continue

            try:
                text = _safe_text(pytesseract.image_to_string(pil_image, config="--psm 6"))
            except Exception:  # noqa: BLE001
                text = ""
            if text:
                chunks.append(text)

        ocr_text = "\n".join(chunks).strip()
        if ocr_text:
            return {
                "text": ocr_text,
                "confidence": 0.72,
                "available": True,
                "engine": "pytesseract",
            }

        return {
            "text": "",
            "confidence": 0.0,
            "available": bool(pytesseract is not None and Image is not None),
            "engine": "pytesseract" if pytesseract is not None else "unavailable",
            "warning": "ocr_no_text_detected",
        }

    def parse(self, profile: DocumentParseProfile) -> dict[str, Any]:
        path = Path(profile.file_path)
        if not path.exists():
            raise FileNotFoundError(f"Parser input file does not exist: {path}")
        if PdfReader is None:
            raise RuntimeError("pypdf is required for pdf-ocr parser adapter")

        max_pages = max(1, _runtime_int(self._config, "documents.pdf_ocr_max_pages", 120))
        timeout_seconds = max(0.0, _runtime_float(self._config, "documents.pdf_ocr_timeout_seconds", 90.0))
        min_text_density = max(0.000001, _runtime_float(self._config, "documents.pdf_ocr_min_text_density", 0.015))
        force_ocr = _runtime_bool(self._config, "documents.pdf_ocr_force", False)

        reader = PdfReader(str(path))
        pages_total = len(reader.pages)
        page_limit = min(pages_total, max_pages)

        warnings: list[str] = []
        errors: list[str] = []
        sections: list[dict[str, Any]] = []
        content_parts: list[str] = []
        page_confidence: list[float] = []
        page_diagnostics: list[dict[str, Any]] = []
        ocr_available_any = False
        ocr_used_pages: list[int] = []
        direct_text_pages: list[int] = []
        parsed_engine_names: set[str] = set()

        if pages_total > page_limit:
            warnings.append(f"pdf_ocr_page_limit_applied:{page_limit}/{pages_total}")

        cursor = 0
        start = time.monotonic()
        for page_index in range(page_limit):
            if timeout_seconds > 0.0 and (time.monotonic() - start) > timeout_seconds:
                warnings.append("pdf_ocr_timeout")
                break

            page_number = page_index + 1
            page = reader.pages[page_index]
            media_box = getattr(page, "mediabox", None)
            try:
                page_width = float(media_box.width) if media_box is not None else 0.0
                page_height = float(media_box.height) if media_box is not None else 0.0
            except Exception:  # noqa: BLE001
                page_width = 0.0
                page_height = 0.0

            try:
                direct_text = _safe_text(page.extract_text() or "")
            except Exception:  # noqa: BLE001
                direct_text = ""

            page_area = max(1.0, page_width * page_height)
            direct_density = float(len(direct_text)) / page_area
            scanned_candidate = not direct_text or direct_density < min_text_density
            should_ocr = force_ocr or scanned_candidate

            page_text = ""
            confidence = 0.0
            ocr_attempted = False
            ocr_used = False
            engine = "none"
            available = False

            if should_ocr:
                ocr_attempted = True
                provider_payload = _coerce_provider_result(
                    self._ocr_provider(
                        page=page,
                        page_number=page_number,
                        direct_text=direct_text,
                        profile=profile,
                    )
                )
                page_text = _safe_text(provider_payload.get("text"))
                confidence = clamp_confidence(provider_payload.get("confidence"), default=0.0)
                engine = _safe_text(provider_payload.get("engine")) or "unknown"
                available = bool(provider_payload.get("available", False))
                warning = _safe_text(provider_payload.get("warning"))
                if warning:
                    warnings.append(warning)
                if page_text:
                    ocr_used = True
                    ocr_used_pages.append(page_number)
                ocr_available_any = ocr_available_any or available
                if engine:
                    parsed_engine_names.add(engine)

            if not page_text and direct_text and not force_ocr:
                page_text = direct_text
                confidence = max(confidence, 0.63)
                direct_text_pages.append(page_number)
                engine = "embedded_text"

            if page_text:
                start_char = cursor
                end_char = start_char + len(page_text)
                sections.append(
                    {
                        "section_id": f"s{len(sections) + 1}",
                        "title": "",
                        "level": 1,
                        "text": page_text,
                        "start_char": start_char,
                        "end_char": end_char,
                        "page_start": page_number,
                        "page_end": page_number,
                        "metadata": {
                            "element_type": "paragraph",
                            "extraction": "ocr" if ocr_used else "text",
                            "ocr_attempted": ocr_attempted,
                            "ocr_used": ocr_used,
                            "ocr_engine": engine,
                            "direct_text_density": direct_density,
                        },
                    }
                )
                content_parts.append(page_text)
                cursor = end_char + 2

            page_confidence.append(confidence)
            page_diagnostics.append(
                {
                    "page_number": page_number,
                    "direct_text_chars": len(direct_text),
                    "direct_text_density": direct_density,
                    "scanned_candidate": scanned_candidate,
                    "ocr_attempted": ocr_attempted,
                    "ocr_used": ocr_used,
                    "ocr_engine": engine,
                    "ocr_available": available,
                    "confidence": confidence,
                }
            )

        content_text = "\n\n".join(part for part in content_parts if part).strip()
        parsed_pages = len(page_diagnostics)
        if parsed_pages == 0:
            status = "failed"
            errors.append("pdf_ocr_no_pages_processed")
            effective_confidence = 0.0
        else:
            status = "parsed" if content_text else "partial"
            average_confidence = sum(page_confidence) / float(len(page_confidence)) if page_confidence else 0.0
            non_empty_pages = len([item for item in page_diagnostics if item.get("confidence", 0.0) > 0.0])
            coverage = non_empty_pages / float(parsed_pages)
            effective_confidence = clamp_confidence(0.35 * coverage + 0.65 * average_confidence, default=0.0)

        if not ocr_available_any and any(item.get("ocr_attempted") for item in page_diagnostics):
            warnings.append("ocr_engine_unavailable")
        if not content_text:
            warnings.append("ocr_no_text_extracted")

        metadata = {
            "parser_kind": "pdf_ocr",
            "pages_total": pages_total,
            "pages_processed": parsed_pages,
            "page_diagnostics": page_diagnostics,
            "ocr_available": ocr_available_any,
            "ocr_engines": sorted(parsed_engine_names),
            "ocr_used_pages": ocr_used_pages,
            "direct_text_pages": direct_text_pages,
            "force_ocr": force_ocr,
        }

        return {
            "status": status,
            "content_text": content_text,
            "sections": sections,
            "metadata": metadata,
            "warnings": warnings,
            "errors": errors,
            "confidence": effective_confidence,
            "cost": clamp_cost(self.cost(profile) + (parsed_pages / 120.0), default=self.cost(profile)),
        }
