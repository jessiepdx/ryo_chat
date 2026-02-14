from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from hypermindlabs.document_parser.adapters.base import (
    DocumentParseProfile,
    DocumentParserAdapter,
    build_parse_profile,
    clamp_confidence,
    clamp_cost,
)
from hypermindlabs.document_parser.contracts import (
    DocumentParserContractError,
    normalize_canonical_parse_output,
)
from hypermindlabs.document_parser.registry import (
    DocumentParserRegistry,
    build_default_parser_registry,
)


class DocumentParserRoutingError(RuntimeError):
    """Raised when parser routing cannot produce a viable adapter chain."""


class DocumentParserExecutionError(RuntimeError):
    """Raised when all routed adapters fail to parse the document."""

    def __init__(self, message: str, *, attempts: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.attempts = list(attempts or [])


class DocumentParserRouter:
    """Deterministic parser routing + fallback chain execution."""

    def __init__(
        self,
        *,
        registry: DocumentParserRegistry | None = None,
        config_manager: Any | None = None,
    ):
        self._config = config_manager
        self._registry = registry if isinstance(registry, DocumentParserRegistry) else build_default_parser_registry(
            config_manager=config_manager
        )

    def _runtime_value(self, path: str, default: Any) -> Any:
        if self._config is None:
            return default
        try:
            return self._config.runtimeValue(path, default)
        except Exception:  # noqa: BLE001
            return default

    def _runtime_int(self, path: str, default: int) -> int:
        try:
            return int(self._runtime_value(path, default))
        except (TypeError, ValueError):
            return int(default)

    def _runtime_float(self, path: str, default: float) -> float:
        try:
            return float(self._runtime_value(path, default))
        except (TypeError, ValueError):
            return float(default)

    def _runtime_bool(self, path: str, default: bool) -> bool:
        value = self._runtime_value(path, default)
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return bool(default)

    def _runtime_csv(self, path: str) -> list[str]:
        value = self._runtime_value(path, [])
        if isinstance(value, list):
            parts = value
        else:
            parts = str(value or "").split(",")
        normalized: list[str] = []
        for item in parts:
            text = str(item).strip().lower()
            if text:
                normalized.append(text)
        return normalized

    def _profile_format_tokens(self, profile: DocumentParseProfile) -> set[str]:
        ext = str(profile.file_extension or "").strip().lower()
        mime = str(profile.file_mime or "").strip().lower()
        tokens: set[str] = set()
        if ext:
            tokens.add(ext)
            tokens.add(ext.lstrip("."))
        if mime:
            tokens.add(mime)
            tokens.add(mime.split("/", 1)[0])
        return tokens

    def _probe_file_content(self, file_path: str) -> dict[str, Any]:
        probe_bytes = max(256, self._runtime_int("documents.parser_probe_bytes", 8192))
        path = Path(str(file_path or "").strip())
        result: dict[str, Any] = {
            "exists": False,
            "pdf_header": False,
            "has_nul": False,
            "utf8_text": False,
            "sample_text": "",
        }
        if not path.exists():
            return result
        try:
            raw = path.read_bytes()[:probe_bytes]
        except OSError:
            return result

        result["exists"] = True
        result["pdf_header"] = raw.startswith(b"%PDF")
        result["has_nul"] = b"\x00" in raw
        try:
            decoded = raw.decode("utf-8")
            result["utf8_text"] = True
        except UnicodeDecodeError:
            decoded = raw.decode("latin-1", errors="ignore")
        result["sample_text"] = " ".join(decoded.replace("\n", " ").split())[:300]
        return result

    def _complexity_from_size(self, file_size_bytes: int) -> tuple[float, str]:
        size = max(0, int(file_size_bytes or 0))
        score = min(1.0, float(size) / float(12 * 1024 * 1024))
        if score >= 0.75:
            return score, "heavy"
        if score >= 0.4:
            return score, "moderate"
        return score, "simple"

    def prepare_profile(self, profile_payload: dict[str, Any]) -> DocumentParseProfile:
        base_profile = build_parse_profile(profile_payload)
        auto_probes = self._probe_file_content(base_profile.file_path)
        merged_probes = dict(auto_probes)
        merged_probes.update(dict(base_profile.probes or {}))
        complexity_score, complexity_label = self._complexity_from_size(base_profile.file_size_bytes)
        if float(base_profile.complexity_score or 0.0) > 0.0:
            complexity_score = float(base_profile.complexity_score)
            complexity_label = str(base_profile.complexity_label or complexity_label)
        return replace(
            base_profile,
            probes=merged_probes,
            complexity_score=min(1.0, max(0.0, complexity_score)),
            complexity_label=str(complexity_label or "simple").lower(),
        )

    def _route_bonus(self, profile: DocumentParseProfile, adapter_name: str) -> float:
        name = str(adapter_name or "").strip().lower()
        mime = str(profile.file_mime or "").strip().lower()
        ext = str(profile.file_extension or "").strip().lower()
        probes = profile.probes if isinstance(profile.probes, dict) else {}
        bonus = 0.0
        if mime == "application/pdf" and name == "pdf-text-layout":
            bonus += 0.25
        if ext == ".pdf" and name == "pdf-text-layout":
            bonus += 0.15
        if probes.get("pdf_header") and name == "pdf-text-layout":
            bonus += 0.2
        if mime == "application/pdf" and name == "pdf-ocr":
            bonus += 0.1
        if ext == ".pdf" and name == "pdf-ocr":
            bonus += 0.05
        if probes.get("pdf_header") and name == "pdf-ocr":
            bonus += 0.05
        if mime == "application/pdf" and name == "pdf-basic":
            bonus += 0.2
        if ext == ".pdf" and name == "pdf-basic":
            bonus += 0.1
        if probes.get("pdf_header") and name == "pdf-basic":
            bonus += 0.2
        if mime in {"text/markdown", "text/x-markdown", "application/markdown"} and name == "text-markdown":
            bonus += 0.25
        if ext in {".md", ".markdown", ".mdown", ".rst", ".txt"} and name == "text-markdown":
            bonus += 0.22
        if mime in {"text/html", "application/xhtml+xml"} and name == "html-web":
            bonus += 0.25
        if ext in {".html", ".htm", ".xhtml"} and name == "html-web":
            bonus += 0.25
        if ext in {".docx", ".pptx", ".xlsx"} and name == "office-suite":
            bonus += 0.28
        if mime in {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        } and name == "office-suite":
            bonus += 0.28
        if ext in {".csv", ".tsv", ".json", ".xml"} and name == "structured-data":
            bonus += 0.25
        if mime in {
            "application/json",
            "text/json",
            "application/xml",
            "text/xml",
            "text/csv",
            "application/csv",
            "text/tab-separated-values",
        } and name == "structured-data":
            bonus += 0.22
        if mime.startswith("text/") and name == "text-plain":
            bonus += 0.2
        if probes.get("utf8_text") and name == "text-plain":
            bonus += 0.1
        if profile.complexity_label == "heavy" and name == "binary-fallback":
            bonus += 0.05
        return bonus

    def route(self, profile: DocumentParseProfile) -> tuple[list[DocumentParserAdapter], dict[str, Any]]:
        preferred = self._runtime_csv("documents.parser_preferred_adapters")
        preferred_rank = {name: index for index, name in enumerate(preferred)}
        max_adapters = max(1, self._runtime_int("documents.parser_max_fallback_adapters", 3))
        cost_ceiling = max(0.1, self._runtime_float("documents.parser_cost_ceiling", 10.0))
        enabled = set(self._runtime_csv("documents.parser_enabled_adapters"))
        format_allowlist = set(self._runtime_csv("documents.parser_format_allowlist"))
        format_denylist = set(self._runtime_csv("documents.parser_format_denylist"))
        format_tokens = self._profile_format_tokens(profile)

        candidates: list[tuple[tuple[Any, ...], DocumentParserAdapter]] = []
        route_debug: dict[str, Any] = {
            "profile": profile.to_dict(),
            "candidates": [],
            "preferred_adapters": preferred,
            "max_adapters": max_adapters,
            "cost_ceiling": cost_ceiling,
            "format_policy": {
                "allowlist": sorted(format_allowlist),
                "denylist": sorted(format_denylist),
                "format_tokens": sorted(format_tokens),
                "blocked_reason": "",
                "blocked_tokens": [],
            },
        }
        if format_allowlist and format_allowlist.isdisjoint(format_tokens):
            route_debug["format_policy"]["blocked_reason"] = "allowlist_miss"
            return [], route_debug
        denied_matches = sorted(token for token in format_tokens if token in format_denylist)
        if denied_matches:
            route_debug["format_policy"]["blocked_reason"] = "denylist_match"
            route_debug["format_policy"]["blocked_tokens"] = denied_matches
            return [], route_debug

        for adapter in self._registry.list_adapters():
            adapter_name = str(getattr(adapter, "adapter_name", "") or "").strip().lower()
            if not adapter_name:
                continue
            if enabled and adapter_name not in enabled:
                route_debug["candidates"].append(
                    {
                        "adapter": adapter_name,
                        "status": "skipped_disabled",
                    }
                )
                continue
            try:
                can_parse = bool(adapter.can_parse(profile))
            except Exception as error:  # noqa: BLE001
                route_debug["candidates"].append(
                    {
                        "adapter": adapter_name,
                        "status": "can_parse_error",
                        "error": str(error),
                    }
                )
                continue
            if not can_parse:
                route_debug["candidates"].append(
                    {
                        "adapter": adapter_name,
                        "status": "not_applicable",
                    }
                )
                continue

            confidence = clamp_confidence(adapter.confidence(profile), default=0.0)
            cost = clamp_cost(adapter.cost(profile), default=0.0)
            score = confidence - (0.08 * cost) + self._route_bonus(profile, adapter_name)
            if cost > cost_ceiling and adapter_name != "binary-fallback":
                route_debug["candidates"].append(
                    {
                        "adapter": adapter_name,
                        "status": "skipped_cost_ceiling",
                        "confidence": confidence,
                        "cost": cost,
                        "score": score,
                    }
                )
                continue
            pref_rank = preferred_rank.get(adapter_name, 9999)
            order_key = (pref_rank, -score, cost, adapter_name)
            candidates.append((order_key, adapter))
            route_debug["candidates"].append(
                {
                    "adapter": adapter_name,
                    "status": "routed",
                    "confidence": confidence,
                    "cost": cost,
                    "score": score,
                    "preferred_rank": pref_rank,
                }
            )

        candidates.sort(key=lambda item: item[0])
        selected: list[DocumentParserAdapter] = []
        seen: set[str] = set()
        for _, adapter in candidates:
            name = str(getattr(adapter, "adapter_name", "") or "").strip().lower()
            if not name or name in seen:
                continue
            seen.add(name)
            selected.append(adapter)
            if len(selected) >= max_adapters:
                break

        route_debug["selected_chain"] = [
            str(getattr(adapter, "adapter_name", "") or "").strip().lower()
            for adapter in selected
        ]
        return selected, route_debug

    def parse_document(self, profile_payload: dict[str, Any]) -> dict[str, Any]:
        if not self._runtime_bool("documents.parser_router_enabled", True):
            raise DocumentParserRoutingError("Parser router is disabled by runtime configuration.")

        profile = self.prepare_profile(profile_payload)
        chain, route_debug = self.route(profile)
        if not chain:
            format_policy = route_debug.get("format_policy") if isinstance(route_debug, dict) else {}
            if isinstance(format_policy, dict) and str(format_policy.get("blocked_reason") or "").strip():
                blocked_reason = str(format_policy.get("blocked_reason") or "policy_blocked")
                blocked_tokens = ",".join(str(item) for item in (format_policy.get("blocked_tokens") or []))
                raise DocumentParserRoutingError(
                    "Document format blocked by parser policy "
                    f"({blocked_reason}): mime='{profile.file_mime}', extension='{profile.file_extension}', "
                    f"matches='{blocked_tokens or 'none'}'."
                )
            raise DocumentParserRoutingError(
                f"No parser adapter matched mime '{profile.file_mime}' and extension '{profile.file_extension}'."
            )

        min_confidence = max(0.0, min(1.0, self._runtime_float("documents.parser_min_confidence", 0.35)))
        fallback_enabled = self._runtime_bool("documents.parser_fallback_enabled", True)
        attempts: list[dict[str, Any]] = []
        attempted_names: list[str] = []

        for index, adapter in enumerate(chain):
            adapter_name = str(getattr(adapter, "adapter_name", "") or "unknown-parser").strip().lower()
            adapter_version = str(getattr(adapter, "adapter_version", "") or "").strip()
            attempted_names.append(adapter_name)
            start_time = time.monotonic()
            try:
                output = adapter.parse(profile)
                duration_ms = max(0, int((time.monotonic() - start_time) * 1000))
                effective_confidence = clamp_confidence(
                    _coerce_dict(output).get("confidence"),
                    default=clamp_confidence(adapter.confidence(profile), default=0.0),
                )
                effective_cost = clamp_cost(
                    _coerce_dict(output).get("cost"),
                    default=clamp_cost(adapter.cost(profile), default=0.0),
                )
                canonical = normalize_canonical_parse_output(
                    output,
                    parser_name=adapter_name,
                    parser_version=adapter_version,
                    adapter_chain=attempted_names,
                    confidence=effective_confidence,
                    cost=effective_cost,
                    duration_ms=duration_ms,
                    route_debug=route_debug,
                    profile_summary={
                        "file_name": profile.file_name,
                        "file_mime": profile.file_mime,
                        "file_extension": profile.file_extension,
                        "file_size_bytes": profile.file_size_bytes,
                        "complexity_label": profile.complexity_label,
                    },
                )
                status = str(canonical.get("status") or "").strip().lower()
                low_confidence = effective_confidence < min_confidence
                if status not in {"parsed", "partial"}:
                    attempts.append(
                        {
                            "adapter": adapter_name,
                            "status": "invalid_status",
                            "parse_status": status,
                            "confidence": effective_confidence,
                            "duration_ms": duration_ms,
                        }
                    )
                    if fallback_enabled and index + 1 < len(chain):
                        continue
                    raise DocumentParserExecutionError(
                        f"Adapter '{adapter_name}' returned invalid status '{status}'.",
                        attempts=attempts,
                    )
                if low_confidence and fallback_enabled and index + 1 < len(chain):
                    attempts.append(
                        {
                            "adapter": adapter_name,
                            "status": "low_confidence_fallback",
                            "confidence": effective_confidence,
                            "minimum_confidence": min_confidence,
                            "duration_ms": duration_ms,
                        }
                    )
                    continue

                canonical_provenance = canonical.get("provenance")
                if isinstance(canonical_provenance, dict):
                    canonical_provenance["fallback_used"] = len(attempted_names) > 1
                    canonical_provenance["attempts"] = list(attempts)
                return canonical

            except DocumentParserContractError as error:
                attempts.append(
                    {
                        "adapter": adapter_name,
                        "status": "contract_error",
                        "error": str(error),
                    }
                )
                if fallback_enabled and index + 1 < len(chain):
                    continue
                raise DocumentParserExecutionError(
                    f"Adapter '{adapter_name}' returned non-canonical output.",
                    attempts=attempts,
                ) from error
            except Exception as error:  # noqa: BLE001
                attempts.append(
                    {
                        "adapter": adapter_name,
                        "status": "execution_error",
                        "error": str(error),
                    }
                )
                if fallback_enabled and index + 1 < len(chain):
                    continue
                raise DocumentParserExecutionError(
                    f"All parser adapters failed. Last adapter: '{adapter_name}'.",
                    attempts=attempts,
                ) from error

        raise DocumentParserExecutionError(
            "Parser fallback chain exhausted without a valid parse result.",
            attempts=attempts,
        )


def _coerce_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}
