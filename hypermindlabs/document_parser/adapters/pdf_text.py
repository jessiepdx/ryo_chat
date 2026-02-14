from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from hypermindlabs.document_parser.adapters.base import (
    DocumentParseProfile,
    DocumentParserAdapter,
    clamp_confidence,
    clamp_cost,
)
from hypermindlabs.document_parser.pdf_layout import (
    build_layout_sections,
    build_page_diagnostics,
    group_pdf_words_into_lines,
)
from hypermindlabs.document_parser.table_extraction import extract_tables_from_pdfplumber_page

try:
    import pdfplumber
except Exception:  # noqa: BLE001
    pdfplumber = None  # type: ignore[assignment]

try:
    from pypdf import PdfReader
except Exception:  # noqa: BLE001
    PdfReader = None  # type: ignore[assignment]


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


def _runtime_int(config_manager: Any | None, path: str, default: int) -> int:
    try:
        return int(_runtime_value(config_manager, path, default))
    except (TypeError, ValueError):
        return int(default)


def _runtime_float(config_manager: Any | None, path: str, default: float) -> float:
    try:
        return float(_runtime_value(config_manager, path, default))
    except (TypeError, ValueError):
        return float(default)


def _safe_text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _synthesize_lines_from_text(text: str) -> list[dict[str, Any]]:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines: list[dict[str, Any]] = []
    top = 72.0
    for line in normalized.split("\n"):
        candidate = line.strip()
        if not candidate:
            top += 12.0
            continue
        width = max(12.0, float(len(candidate) * 5.5))
        lines.append(
            {
                "text": candidate,
                "x0": 72.0,
                "x1": 72.0 + width,
                "top": top,
                "bottom": top + 12.0,
                "font_size": 11.0,
                "word_count": len(candidate.split(" ")),
            }
        )
        top += 14.0
    return lines


def _compute_parse_confidence(
    *,
    diagnostics: list[dict[str, Any]],
    min_text_density: float,
    timed_out: bool,
    has_content: bool,
) -> float:
    pages = len(diagnostics)
    if pages <= 0:
        return 0.0
    text_pages = len([item for item in diagnostics if item.get("has_extractable_text")])
    scanned_pages = len([item for item in diagnostics if item.get("scanned_candidate")])
    coverage = text_pages / float(pages)
    scanned_ratio = scanned_pages / float(pages)
    avg_density = sum(float(item.get("text_density", 0.0) or 0.0) for item in diagnostics) / float(pages)

    score = 0.2 + (0.55 * coverage)
    if avg_density >= max(0.00001, float(min_text_density)) * 2.0:
        score += 0.15
    elif avg_density < max(0.00001, float(min_text_density)):
        score -= 0.12
    score -= 0.42 * scanned_ratio
    if timed_out:
        score -= 0.18
    if not has_content:
        score -= 0.25
    return clamp_confidence(score, default=0.0)


def _parse_with_pdfplumber(
    path: Path,
    *,
    max_pages: int,
    timeout_seconds: float,
    layout_enabled: bool,
    table_extraction_enabled: bool,
    table_max_per_page: int,
    table_min_cells: int,
) -> tuple[list[dict[str, Any]], int, bool, list[str]]:
    page_layouts: list[dict[str, Any]] = []
    warnings: list[str] = []
    timed_out = False
    pages_total = 0
    start = time.monotonic()

    if pdfplumber is None:
        return page_layouts, pages_total, timed_out, ["pdfplumber_unavailable"]

    with pdfplumber.open(str(path)) as pdf_doc:
        pages_total = len(pdf_doc.pages)
        page_limit = pages_total if max_pages <= 0 else min(pages_total, max_pages)
        if pages_total > page_limit:
            warnings.append(f"pdf_page_limit_applied:{page_limit}/{pages_total}")

        for page_index in range(page_limit):
            if timeout_seconds > 0.0 and (time.monotonic() - start) > timeout_seconds:
                warnings.append("pdf_text_parse_timeout")
                timed_out = True
                break
            page_number = page_index + 1
            page = pdf_doc.pages[page_index]
            page_width = float(getattr(page, "width", 0.0) or 0.0)
            page_height = float(getattr(page, "height", 0.0) or 0.0)

            words: list[dict[str, Any]] = []
            lines: list[dict[str, Any]] = []
            page_text = ""

            if layout_enabled:
                try:
                    words = list(page.extract_words(use_text_flow=True, keep_blank_chars=False) or [])
                except TypeError:
                    words = list(page.extract_words() or [])
                except Exception:  # noqa: BLE001
                    words = []
                lines = group_pdf_words_into_lines(words)
                page_text = "\n".join(_safe_text(line.get("text")) for line in lines if _safe_text(line.get("text")))

            if not page_text:
                try:
                    page_text = _safe_text(page.extract_text(layout=True))
                except TypeError:
                    page_text = _safe_text(page.extract_text())
                except Exception:  # noqa: BLE001
                    page_text = ""
                if page_text and not lines:
                    lines = _synthesize_lines_from_text(page_text)

            tables: list[dict[str, Any]] = []
            if table_extraction_enabled:
                tables = extract_tables_from_pdfplumber_page(
                    page,
                    page_number=page_number,
                    max_tables=table_max_per_page,
                    min_cells=table_min_cells,
                )

            page_layouts.append(
                {
                    "page_number": page_number,
                    "page_width": page_width,
                    "page_height": page_height,
                    "lines": lines,
                    "tables": tables,
                    "raw_text": page_text,
                }
            )

    return page_layouts, pages_total, timed_out, warnings


def _parse_with_pypdf(
    path: Path,
    *,
    max_pages: int,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], int, bool, list[str]]:
    page_layouts: list[dict[str, Any]] = []
    warnings: list[str] = []
    timed_out = False
    pages_total = 0
    start = time.monotonic()

    if PdfReader is None:
        return page_layouts, pages_total, timed_out, ["pypdf_unavailable"]

    reader = PdfReader(str(path))
    pages_total = len(reader.pages)
    page_limit = pages_total if max_pages <= 0 else min(pages_total, max_pages)
    if pages_total > page_limit:
        warnings.append(f"pdf_page_limit_applied:{page_limit}/{pages_total}")

    for page_index in range(page_limit):
        if timeout_seconds > 0.0 and (time.monotonic() - start) > timeout_seconds:
            warnings.append("pdf_text_parse_timeout")
            timed_out = True
            break
        page_number = page_index + 1
        page = reader.pages[page_index]
        try:
            page_text = _safe_text(page.extract_text() or "")
        except Exception:  # noqa: BLE001
            page_text = ""
        lines = _synthesize_lines_from_text(page_text)
        media_box = getattr(page, "mediabox", None)
        width = 0.0
        height = 0.0
        try:
            width = float(media_box.width) if media_box is not None else 0.0
            height = float(media_box.height) if media_box is not None else 0.0
        except Exception:  # noqa: BLE001
            width = 0.0
            height = 0.0
        page_layouts.append(
            {
                "page_number": page_number,
                "page_width": width,
                "page_height": height,
                "lines": lines,
                "tables": [],
                "raw_text": page_text,
            }
        )

    warnings.append("pdf_layout_fallback_pypdf")
    return page_layouts, pages_total, timed_out, warnings


class PdfTextLayoutParserAdapter(DocumentParserAdapter):
    adapter_name = "pdf-text-layout"
    adapter_version = "v1"

    def __init__(self, *, config_manager: Any | None = None):
        self._config = config_manager

    def can_parse(self, profile: DocumentParseProfile) -> bool:
        mime = str(profile.file_mime or "").strip().lower()
        probes = profile.probes if isinstance(profile.probes, dict) else {}
        if mime == "application/pdf":
            return True
        if profile.file_extension == ".pdf":
            return True
        return bool(probes.get("pdf_header"))

    def confidence(self, profile: DocumentParseProfile) -> float:
        mime = str(profile.file_mime or "").strip().lower()
        probes = profile.probes if isinstance(profile.probes, dict) else {}
        score = 0.35
        if mime == "application/pdf":
            score += 0.3
        if profile.file_extension == ".pdf":
            score += 0.2
        if probes.get("pdf_header"):
            score += 0.15
        score -= max(0.0, float(profile.complexity_score or 0.0) - 0.75) * 0.25
        return clamp_confidence(score, default=0.0)

    def cost(self, profile: DocumentParseProfile) -> float:
        size_mb = max(0.0, float(profile.file_size_bytes or 0) / (1024.0 * 1024.0))
        return clamp_cost(1.8 + (size_mb / 2.0), default=1.8)

    def parse(self, profile: DocumentParseProfile) -> dict[str, Any]:
        path = Path(profile.file_path)
        if not path.exists():
            raise FileNotFoundError(f"Parser input file does not exist: {path}")

        max_pages = max(1, _runtime_int(self._config, "documents.pdf_max_pages", 500))
        timeout_seconds = max(0.0, _runtime_float(self._config, "documents.pdf_text_timeout_seconds", 30.0))
        layout_enabled = _runtime_bool(self._config, "documents.pdf_layout_enabled", True)
        table_extraction_enabled = _runtime_bool(self._config, "documents.pdf_table_extraction_enabled", True)
        table_max_per_page = max(0, _runtime_int(self._config, "documents.pdf_table_max_per_page", 6))
        table_min_cells = max(1, _runtime_int(self._config, "documents.pdf_table_min_cells", 4))
        min_text_density = max(0.000001, _runtime_float(self._config, "documents.pdf_ocr_min_text_density", 0.015))
        ocr_recommend_ratio = max(
            0.0,
            min(1.0, _runtime_float(self._config, "documents.pdf_ocr_min_scanned_page_ratio", 0.35)),
        )
        ocr_enabled = _runtime_bool(self._config, "documents.pdf_ocr_enabled", True)

        warnings: list[str] = []
        parse_errors: list[str] = []
        page_layouts: list[dict[str, Any]] = []
        pages_total = 0
        timed_out = False

        if pdfplumber is not None:
            try:
                page_layouts, pages_total, timed_out, parse_warnings = _parse_with_pdfplumber(
                    path,
                    max_pages=max_pages,
                    timeout_seconds=timeout_seconds,
                    layout_enabled=layout_enabled,
                    table_extraction_enabled=table_extraction_enabled,
                    table_max_per_page=table_max_per_page,
                    table_min_cells=table_min_cells,
                )
                warnings.extend(parse_warnings)
            except Exception as error:  # noqa: BLE001
                parse_errors.append(f"pdfplumber_parse_error:{error}")

        if not page_layouts:
            fallback_layouts, pages_total, timed_out, fallback_warnings = _parse_with_pypdf(
                path,
                max_pages=max_pages,
                timeout_seconds=timeout_seconds,
            )
            page_layouts = fallback_layouts
            warnings.extend(fallback_warnings)

        diagnostics = build_page_diagnostics(page_layouts, min_text_density=min_text_density)
        sections, content_text, elements = build_layout_sections(page_layouts)
        if not sections and content_text:
            sections = [
                {
                    "section_id": "s1",
                    "title": "",
                    "level": 1,
                    "text": content_text,
                    "start_char": 0,
                    "end_char": len(content_text),
                    "page_start": 1,
                    "page_end": max(1, len(page_layouts)),
                    "metadata": {"element_type": "paragraph", "source": "pdf_text_layout_fallback"},
                }
            ]

        scanned_pages = [
            int(item.get("page_number") or 0)
            for item in diagnostics
            if bool(item.get("scanned_candidate"))
        ]
        processed_pages = len(diagnostics)
        scanned_ratio = (len(scanned_pages) / float(processed_pages)) if processed_pages else 1.0
        has_content = bool(content_text.strip())

        if not has_content:
            warnings.append("no_extractable_pdf_text")
        if timed_out:
            warnings.append("partial_parse_timeout")
        if ocr_enabled and scanned_ratio >= ocr_recommend_ratio and processed_pages > 0:
            warnings.append("ocr_recommended_for_scanned_pages")

        confidence = _compute_parse_confidence(
            diagnostics=diagnostics,
            min_text_density=min_text_density,
            timed_out=timed_out,
            has_content=has_content,
        )
        estimated_cost = clamp_cost(
            self.cost(profile) + (float(processed_pages) / 250.0),
            default=self.cost(profile),
        )

        status = "parsed" if has_content and not timed_out else "partial"

        metadata = {
            "parser_kind": "pdf_text_layout",
            "pages_total": pages_total,
            "pages_processed": processed_pages,
            "layout_enabled": layout_enabled,
            "table_extraction_enabled": table_extraction_enabled,
            "timed_out": timed_out,
            "scanned_page_numbers": scanned_pages,
            "scanned_page_ratio": scanned_ratio,
            "page_diagnostics": diagnostics,
            "element_count": len(elements),
        }
        if parse_errors:
            metadata["parse_errors"] = parse_errors

        return {
            "status": status,
            "content_text": content_text,
            "sections": sections,
            "metadata": metadata,
            "warnings": warnings,
            "errors": parse_errors,
            "confidence": confidence,
            "cost": estimated_cost,
        }
