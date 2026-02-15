from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


CONTROLLED_TOPIC_LABELS: tuple[str, ...] = (
    "api",
    "architecture",
    "authentication",
    "authorization",
    "automation",
    "billing",
    "checklist",
    "compliance",
    "contract",
    "data",
    "deployment",
    "faq",
    "finance",
    "governance",
    "hr",
    "incident",
    "integration",
    "legal",
    "monitoring",
    "onboarding",
    "operations",
    "performance",
    "policy",
    "pricing",
    "privacy",
    "procedure",
    "procurement",
    "product",
    "quality",
    "release",
    "reliability",
    "reporting",
    "retention",
    "risk",
    "roadmap",
    "security",
    "sla",
    "support",
    "template",
    "testing",
    "troubleshooting",
)

CONTROLLED_FORMAT_LABELS: tuple[str, ...] = (
    "code",
    "faq",
    "narrative",
    "policy",
    "procedure",
    "reference",
    "table",
    "checklist",
)

CONTROLLED_DOMAIN_LABELS: tuple[str, ...] = (
    "engineering",
    "finance",
    "general",
    "legal",
    "operations",
    "product",
    "security",
    "support",
)

_STOPWORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "with",
}

_TOPIC_SYNONYM_GROUPS: dict[str, tuple[str, ...]] = {
    "api": ("api", "apis", "endpoint", "endpoints"),
    "architecture": ("architecture", "design", "system design"),
    "authentication": ("authentication", "authn", "signin", "login"),
    "authorization": ("authorization", "authz", "permission", "permissions", "rbac"),
    "automation": ("automation", "automated", "workflow automation"),
    "billing": ("billing", "invoice", "invoicing"),
    "checklist": ("checklist", "todo", "to-do"),
    "compliance": ("compliance", "regulatory", "regulation", "regulated"),
    "contract": ("contract", "agreement", "msa", "sow"),
    "data": ("data", "dataset", "schema"),
    "deployment": ("deployment", "deploy", "release deploy"),
    "faq": ("faq", "q&a", "question answer"),
    "finance": ("finance", "financial", "budget"),
    "governance": ("governance", "governing"),
    "hr": ("hr", "human resources"),
    "incident": ("incident", "outage", "sev", "postmortem"),
    "integration": ("integration", "integrations", "connector", "connectors"),
    "legal": ("legal", "counsel", "law"),
    "monitoring": ("monitoring", "observability", "alerting"),
    "onboarding": ("onboarding", "on-board", "new hire"),
    "operations": ("operations", "ops", "runops"),
    "performance": ("performance", "latency", "throughput"),
    "policy": ("policy", "policies", "governance policy"),
    "pricing": ("pricing", "price", "rate card"),
    "privacy": ("privacy", "pii", "gdpr"),
    "procedure": ("procedure", "procedures", "runbook", "sop"),
    "procurement": ("procurement", "vendor", "purchasing"),
    "product": ("product", "feature", "roadmap feature"),
    "quality": ("quality", "qa", "assurance"),
    "release": ("release", "changelog", "release note"),
    "reliability": ("reliability", "uptime", "availability"),
    "reporting": ("reporting", "report", "dashboard"),
    "retention": ("retention", "retained", "archive", "archival"),
    "risk": ("risk", "threat", "mitigation"),
    "roadmap": ("roadmap", "milestone", "timeline"),
    "security": ("security", "vulnerability", "threat model"),
    "sla": ("sla", "service level"),
    "support": ("support", "helpdesk", "ticket"),
    "template": ("template", "boilerplate"),
    "testing": ("testing", "test", "unit test", "integration test"),
    "troubleshooting": ("troubleshooting", "debug", "diagnostics"),
}

_FORMAT_SYNONYM_GROUPS: dict[str, tuple[str, ...]] = {
    "code": ("code", "snippet", "function", "class"),
    "faq": ("faq", "q&a", "question"),
    "narrative": ("overview", "introduction", "summary"),
    "policy": ("policy", "must", "shall", "required"),
    "procedure": ("procedure", "step", "runbook", "sop"),
    "reference": ("reference", "appendix"),
    "table": ("table", "row", "column"),
    "checklist": ("checklist", "todo", "to-do"),
}

_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "engineering": ("api", "architecture", "code", "deployment", "integration", "testing"),
    "finance": ("billing", "pricing", "budget", "finance", "invoice"),
    "legal": ("contract", "compliance", "legal", "privacy", "retention", "policy"),
    "operations": ("incident", "monitoring", "operations", "procedure", "runbook", "reliability"),
    "product": ("product", "roadmap", "release", "feature"),
    "security": ("security", "authentication", "authorization", "risk", "vulnerability", "pii"),
    "support": ("support", "faq", "troubleshooting", "ticket"),
}

_TOPIC_TO_DOMAIN: dict[str, tuple[str, ...]] = {
    "api": ("engineering",),
    "architecture": ("engineering",),
    "authentication": ("security", "engineering"),
    "authorization": ("security", "engineering"),
    "billing": ("finance",),
    "compliance": ("legal", "security"),
    "contract": ("legal",),
    "deployment": ("engineering", "operations"),
    "faq": ("support",),
    "incident": ("operations", "security"),
    "monitoring": ("operations",),
    "performance": ("engineering", "operations"),
    "policy": ("legal", "operations"),
    "pricing": ("finance", "product"),
    "privacy": ("legal", "security"),
    "procedure": ("operations",),
    "release": ("product", "engineering"),
    "reliability": ("operations", "engineering"),
    "retention": ("legal", "operations"),
    "risk": ("security", "legal"),
    "roadmap": ("product",),
    "security": ("security",),
    "sla": ("support", "operations"),
    "support": ("support",),
    "testing": ("engineering", "quality"),
    "troubleshooting": ("support", "operations"),
}

_TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9_-]{2,}")


def _as_text(value: Any, default: str = "") -> str:
    text = str(value if value is not None else "").strip()
    return text if text else default


def _normalize_label(value: str) -> str:
    text = _as_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text


def _build_synonym_index(groups: dict[str, tuple[str, ...]]) -> dict[str, str]:
    index: dict[str, str] = {}
    for canonical, aliases in groups.items():
        canonical_key = _normalize_label(canonical)
        if canonical_key:
            index[canonical_key] = canonical
        for alias in aliases:
            alias_key = _normalize_label(alias)
            if alias_key:
                index[alias_key] = canonical
    return index


_TOPIC_SYNONYM_INDEX = _build_synonym_index(_TOPIC_SYNONYM_GROUPS)
_FORMAT_SYNONYM_INDEX = _build_synonym_index(_FORMAT_SYNONYM_GROUPS)
_TOPIC_PHRASES = sorted(
    [key for key in _TOPIC_SYNONYM_INDEX.keys() if " " in key],
    key=lambda item: (-len(item), item),
)
_FORMAT_PHRASES = sorted(
    [key for key in _FORMAT_SYNONYM_INDEX.keys() if " " in key],
    key=lambda item: (-len(item), item),
)


def normalize_topic_label(label: Any) -> str:
    normalized = _normalize_label(_as_text(label))
    if not normalized:
        return ""
    canonical = _TOPIC_SYNONYM_INDEX.get(normalized, normalized)
    if canonical in CONTROLLED_TOPIC_LABELS:
        return canonical
    return ""


def normalize_format_label(label: Any) -> str:
    normalized = _normalize_label(_as_text(label))
    if not normalized:
        return ""
    canonical = _FORMAT_SYNONYM_INDEX.get(normalized, normalized)
    if canonical in CONTROLLED_FORMAT_LABELS:
        return canonical
    return ""


def normalize_domain_label(label: Any) -> str:
    normalized = _normalize_label(_as_text(label))
    if normalized in CONTROLLED_DOMAIN_LABELS:
        return normalized
    if normalized == "ops":
        return "operations"
    if normalized == "eng":
        return "engineering"
    return ""


def _rank_signals(
    scores: dict[str, float],
    *,
    min_confidence: float,
    max_items: int,
) -> list[dict[str, Any]]:
    if not scores:
        return []
    max_score = max(0.000001, max(float(score) for score in scores.values()))
    ranked: list[dict[str, Any]] = []
    for label, score in scores.items():
        confidence = min(1.0, max(0.0, float(score) / max_score))
        if confidence < max(0.0, float(min_confidence)):
            continue
        ranked.append(
            {
                "label": label,
                "confidence": round(confidence, 4),
            }
        )
    ranked.sort(key=lambda item: (-float(item["confidence"]), str(item["label"])))
    return ranked[: max(1, int(max_items))]


def flatten_signal_labels(signals: list[dict[str, Any]] | None, *, max_items: int = 8) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in list(signals or []):
        label = _as_text((item or {}).get("label")).lower()
        if not label or label in seen:
            continue
        seen.add(label)
        output.append(label)
        if len(output) >= max(1, int(max_items)):
            break
    return output


def extract_topic_signals(
    text: str,
    *,
    max_topics: int = 8,
    min_confidence: float = 0.2,
    synonym_expansion: bool = True,
) -> list[dict[str, Any]]:
    normalized_text = _normalize_label(text)
    if not normalized_text:
        return []

    scores: dict[str, float] = defaultdict(float)
    if synonym_expansion:
        for phrase in _TOPIC_PHRASES:
            if phrase in normalized_text:
                canonical = _TOPIC_SYNONYM_INDEX.get(phrase)
                if canonical:
                    scores[canonical] += 1.4

    for token in _TOKEN_PATTERN.findall(normalized_text):
        if token in _STOPWORDS:
            continue
        canonical = _TOPIC_SYNONYM_INDEX.get(token, token if token in CONTROLLED_TOPIC_LABELS else "")
        if not canonical:
            continue
        scores[canonical] += 1.0

    return _rank_signals(scores, min_confidence=min_confidence, max_items=max_topics)


def detect_format_signals(
    text: str,
    *,
    node_type: str = "",
    max_labels: int = 4,
    min_confidence: float = 0.2,
) -> list[dict[str, Any]]:
    normalized_text = _normalize_label(text)
    scores: dict[str, float] = defaultdict(float)

    node_type_label = _normalize_label(node_type)
    if node_type_label == "table":
        scores["table"] = max(scores["table"], 2.0)
    elif node_type_label == "code":
        scores["code"] = max(scores["code"], 2.0)
    elif node_type_label == "list":
        scores["checklist"] = max(scores["checklist"], 1.5)
    elif node_type_label in {"section", "subsection", "paragraph"}:
        scores["narrative"] = max(scores["narrative"], 0.7)

    if "\n" in str(text) and "|" in str(text):
        scores["table"] = max(scores["table"], 1.2)
    if "```" in str(text) or "def " in str(text) or "class " in str(text):
        scores["code"] = max(scores["code"], 1.3)
    if str(text).count("?") >= 2 or "faq" in normalized_text:
        scores["faq"] = max(scores["faq"], 1.2)
    if any(token in normalized_text for token in ("must ", " shall ", "required", "policy")):
        scores["policy"] = max(scores["policy"], 1.2)
    if any(token in normalized_text for token in ("step ", "procedure", "runbook", "sop")):
        scores["procedure"] = max(scores["procedure"], 1.1)
    if re.search(r"(^|\n)\s*(?:-|\*|\d+[.)])\s+", str(text)):
        scores["checklist"] = max(scores["checklist"], 1.0)

    for phrase in _FORMAT_PHRASES:
        if phrase in normalized_text:
            canonical = _FORMAT_SYNONYM_INDEX.get(phrase)
            if canonical:
                scores[canonical] = max(scores[canonical], 1.0)

    ranked = _rank_signals(scores, min_confidence=min_confidence, max_items=max_labels)
    normalized: list[dict[str, Any]] = []
    for item in ranked:
        label = normalize_format_label(item.get("label"))
        if not label:
            continue
        normalized.append({"label": label, "confidence": float(item.get("confidence", 0.0))})
    normalized.sort(key=lambda item: (-float(item["confidence"]), str(item["label"])))
    return normalized[: max(1, int(max_labels))]


def detect_domain_signals(
    text: str,
    *,
    topic_signals: list[dict[str, Any]] | None = None,
    max_domains: int = 3,
    min_confidence: float = 0.2,
) -> list[dict[str, Any]]:
    normalized_text = _normalize_label(text)
    tokens = set(_TOKEN_PATTERN.findall(normalized_text))
    scores: dict[str, float] = defaultdict(float)
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        keyword_hits = len([keyword for keyword in keywords if keyword in tokens or keyword in normalized_text])
        if keyword_hits > 0:
            scores[domain] += float(keyword_hits)

    for signal in list(topic_signals or []):
        topic_label = normalize_topic_label((signal or {}).get("label"))
        confidence = max(0.0, min(1.0, float((signal or {}).get("confidence", 0.0) or 0.0)))
        for domain in _TOPIC_TO_DOMAIN.get(topic_label, tuple()):
            scores[domain] += max(0.1, confidence)

    if not scores:
        return [{"label": "general", "confidence": 0.5}]
    ranked = _rank_signals(scores, min_confidence=min_confidence, max_items=max_domains)
    normalized: list[dict[str, Any]] = []
    for item in ranked:
        label = normalize_domain_label(item.get("label"))
        if not label:
            continue
        normalized.append({"label": label, "confidence": float(item.get("confidence", 0.0))})
    if not normalized:
        normalized.append({"label": "general", "confidence": 0.5})
    normalized.sort(key=lambda item: (-float(item["confidence"]), str(item["label"])))
    return normalized[: max(1, int(max_domains))]
