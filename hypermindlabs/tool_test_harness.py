from __future__ import annotations

import copy
import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_TOOL_TEST_HARNESS_PATH = (
    Path(__file__).resolve().parent.parent / "db" / "tool_test_harness.json"
)
MAX_RUN_HISTORY = 25
MAX_DIFF_ITEMS = 64


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_text(value: Any, fallback: str = "") -> str:
    text = str(value if value is not None else "").strip()
    return text if text else fallback


def _as_bool(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(fallback)


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    return {}


def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return copy.deepcopy(value)
    return []


def _json_safe(value: Any, depth: int = 0) -> Any:
    if depth > 12:
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, nested in value.items():
            output[str(key)] = _json_safe(nested, depth + 1)
        return output
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, depth + 1) for item in value]
    return str(value)


def _json_hash(value: Any) -> str:
    canonical = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _preview(value: Any, max_chars: int = 180) -> str:
    text = json.dumps(_json_safe(value), ensure_ascii=True, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return f"{text[: max(0, max_chars - 3)]}..."


def _schema_type(raw_type: Any) -> list[str]:
    if isinstance(raw_type, str):
        return [raw_type]
    if isinstance(raw_type, list):
        values = [str(item).strip() for item in raw_type if str(item).strip()]
        return sorted(set(values))
    return []


def _safe_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    payload = _coerce_dict(schema)
    if payload.get("type") != "object":
        payload["type"] = "object"
    properties = _coerce_dict(payload.get("properties"))
    payload["properties"] = properties
    required = []
    for item in _coerce_list(payload.get("required")):
        clean = str(item).strip()
        if clean:
            required.append(clean)
    payload["required"] = sorted(set(required))
    return payload


def build_contract_snapshot(tool_name: str, input_schema: dict[str, Any] | None) -> dict[str, Any]:
    schema = _safe_schema(input_schema)
    properties = _coerce_dict(schema.get("properties"))
    property_contract: dict[str, Any] = {}

    for prop_name in sorted(properties.keys()):
        prop_schema = _coerce_dict(properties.get(prop_name))
        prop_entry: dict[str, Any] = {
            "type": _schema_type(prop_schema.get("type")),
        }
        enum_values = _coerce_list(prop_schema.get("enum"))
        if enum_values:
            prop_entry["enum"] = [_json_safe(item) for item in enum_values]
        format_name = _as_text(prop_schema.get("format"))
        if format_name:
            prop_entry["format"] = format_name
        property_contract[prop_name] = prop_entry

    contract: dict[str, Any] = {
        "schema": "ryo.tool_contract.v1",
        "tool_name": _as_text(tool_name),
        "required": _coerce_list(schema.get("required")),
        "properties": property_contract,
    }
    contract["hash"] = _json_hash(contract)
    return contract


def compare_contract_snapshots(
    expected_contract: dict[str, Any] | None,
    current_contract: dict[str, Any] | None,
) -> dict[str, Any]:
    expected = _coerce_dict(expected_contract)
    current = _coerce_dict(current_contract)
    if not expected:
        return {
            "status": "missing_baseline",
            "compatible": True,
            "message": "No contract baseline saved yet.",
            "expected_hash": None,
            "current_hash": _as_text(current.get("hash")) or _json_hash(current),
            "added_required": [],
            "removed_required": [],
            "added_properties": [],
            "removed_properties": [],
            "type_changes": [],
        }

    expected_required = set(str(item).strip() for item in _coerce_list(expected.get("required")) if str(item).strip())
    current_required = set(str(item).strip() for item in _coerce_list(current.get("required")) if str(item).strip())
    expected_props = _coerce_dict(expected.get("properties"))
    current_props = _coerce_dict(current.get("properties"))

    added_required = sorted(current_required - expected_required)
    removed_required = sorted(expected_required - current_required)
    added_properties = sorted(set(current_props.keys()) - set(expected_props.keys()))
    removed_properties = sorted(set(expected_props.keys()) - set(current_props.keys()))

    type_changes: list[dict[str, Any]] = []
    for prop_name in sorted(set(expected_props.keys()) & set(current_props.keys())):
        expected_types = sorted(_schema_type(_coerce_dict(expected_props[prop_name]).get("type")))
        current_types = sorted(_schema_type(_coerce_dict(current_props[prop_name]).get("type")))
        if expected_types != current_types:
            type_changes.append(
                {
                    "property": prop_name,
                    "expected": expected_types,
                    "actual": current_types,
                }
            )

    compatible = not (added_required or removed_required or removed_properties or type_changes)
    return {
        "status": "pass" if compatible else "fail",
        "compatible": compatible,
        "message": "Contract compatible." if compatible else "Contract drift detected.",
        "expected_hash": _as_text(expected.get("hash")) or _json_hash(expected),
        "current_hash": _as_text(current.get("hash")) or _json_hash(current),
        "added_required": added_required,
        "removed_required": removed_required,
        "added_properties": added_properties,
        "removed_properties": removed_properties,
        "type_changes": type_changes,
    }


def _diff_values(expected: Any, actual: Any, path: str = "$", diff_limit: int = MAX_DIFF_ITEMS) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []

    def walk(left: Any, right: Any, current_path: str) -> None:
        if len(diffs) >= diff_limit:
            return

        left_safe = _json_safe(left)
        right_safe = _json_safe(right)

        if type(left_safe) is not type(right_safe):
            diffs.append(
                {
                    "path": current_path,
                    "kind": "type_mismatch",
                    "expected": _preview(left_safe),
                    "actual": _preview(right_safe),
                }
            )
            return

        if isinstance(left_safe, dict):
            left_keys = set(left_safe.keys())
            right_keys = set(right_safe.keys())
            for missing_key in sorted(left_keys - right_keys):
                if len(diffs) >= diff_limit:
                    return
                diffs.append(
                    {
                        "path": f"{current_path}.{missing_key}",
                        "kind": "missing_key",
                        "expected": _preview(left_safe.get(missing_key)),
                        "actual": "<missing>",
                    }
                )
            for extra_key in sorted(right_keys - left_keys):
                if len(diffs) >= diff_limit:
                    return
                diffs.append(
                    {
                        "path": f"{current_path}.{extra_key}",
                        "kind": "unexpected_key",
                        "expected": "<missing>",
                        "actual": _preview(right_safe.get(extra_key)),
                    }
                )
            for common_key in sorted(left_keys & right_keys):
                walk(left_safe.get(common_key), right_safe.get(common_key), f"{current_path}.{common_key}")
            return

        if isinstance(left_safe, list):
            if len(left_safe) != len(right_safe):
                diffs.append(
                    {
                        "path": current_path,
                        "kind": "length_mismatch",
                        "expected": len(left_safe),
                        "actual": len(right_safe),
                    }
                )
            for index, (left_item, right_item) in enumerate(zip(left_safe, right_safe)):
                if len(diffs) >= diff_limit:
                    return
                walk(left_item, right_item, f"{current_path}[{index}]")
            return

        if left_safe != right_safe:
            diffs.append(
                {
                    "path": current_path,
                    "kind": "value_mismatch",
                    "expected": _preview(left_safe),
                    "actual": _preview(right_safe),
                }
            )

    walk(expected, actual, path)
    return diffs


def compare_golden_outputs(expected_output: Any, actual_output: Any) -> dict[str, Any]:
    if expected_output is None:
        return {
            "status": "missing_golden",
            "match": None,
            "message": "Golden output not set.",
            "expected_hash": None,
            "actual_hash": _json_hash(actual_output),
            "diff_count": 0,
            "diffs": [],
        }

    expected_safe = _json_safe(expected_output)
    actual_safe = _json_safe(actual_output)
    expected_hash = _json_hash(expected_safe)
    actual_hash = _json_hash(actual_safe)
    if expected_hash == actual_hash:
        return {
            "status": "pass",
            "match": True,
            "message": "Golden output matches.",
            "expected_hash": expected_hash,
            "actual_hash": actual_hash,
            "diff_count": 0,
            "diffs": [],
        }

    diffs = _diff_values(expected_safe, actual_safe)
    return {
        "status": "fail",
        "match": False,
        "message": "Golden output drift detected.",
        "expected_hash": expected_hash,
        "actual_hash": actual_hash,
        "diff_count": len(diffs),
        "diffs": diffs,
    }


class ToolHarnessValidationError(ValueError):
    """Raised for invalid tool harness payloads."""


class ToolTestHarnessStore:
    """File-backed storage for tool fixture cases and regression reports."""

    def __init__(self, storage_path: str | Path = DEFAULT_TOOL_TEST_HARNESS_PATH):
        self._storage_path = Path(storage_path)
        self._lock = threading.RLock()
        self._ensure_store()

    def _ensure_store(self) -> None:
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        if self._storage_path.exists():
            return
        payload = {
            "schema": "ryo.tool_harness.v1",
            "updated_at": _now_iso(),
            "cases": {},
        }
        self._storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load(self) -> dict[str, Any]:
        self._ensure_store()
        try:
            raw = json.loads(self._storage_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            raw = {}
        payload = _coerce_dict(raw)
        cases = payload.get("cases")
        if not isinstance(cases, dict):
            cases = {}
        return {
            "schema": "ryo.tool_harness.v1",
            "updated_at": _as_text(payload.get("updated_at")),
            "cases": cases,
        }

    def _save(self, payload: dict[str, Any]) -> None:
        payload["updated_at"] = _now_iso()
        self._storage_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")

    @staticmethod
    def _normalize_case(
        payload: dict[str, Any] | None,
        *,
        existing: dict[str, Any] | None = None,
        actor_member_id: int = 0,
    ) -> dict[str, Any]:
        data = _coerce_dict(payload)
        current = _coerce_dict(existing)
        if not data and not current:
            raise ToolHarnessValidationError("Tool harness case payload cannot be empty.")

        case_id = _as_text(data.get("case_id"), _as_text(current.get("case_id")))
        if not case_id:
            case_id = f"case-{uuid.uuid4().hex[:16]}"

        tool_name = _as_text(data.get("tool_name"), _as_text(current.get("tool_name")))
        if not tool_name:
            raise ToolHarnessValidationError("tool_name is required.")

        fixture_name = _as_text(data.get("fixture_name"), _as_text(current.get("fixture_name"), f"{tool_name} fixture"))
        description = _as_text(data.get("description"), _as_text(current.get("description")))
        execution_mode = _as_text(data.get("execution_mode"), _as_text(current.get("execution_mode"), "real")).lower()
        if execution_mode not in {"real", "mock"}:
            execution_mode = "real"

        input_args: dict[str, Any]
        if "input_args" in data:
            raw_args = data.get("input_args")
            if isinstance(raw_args, str):
                try:
                    parsed = json.loads(raw_args)
                except json.JSONDecodeError as error:
                    raise ToolHarnessValidationError("input_args must be valid JSON object.") from error
                input_args = _coerce_dict(parsed)
            else:
                input_args = _coerce_dict(raw_args)
        else:
            input_args = _coerce_dict(current.get("input_args"))

        tags_source = data.get("tags") if "tags" in data else current.get("tags")
        tags = [str(item).strip() for item in _coerce_list(tags_source) if str(item).strip()]
        enabled = _as_bool(data.get("enabled"), _as_bool(current.get("enabled"), True))

        if "contract_snapshot" in data:
            contract_snapshot = _coerce_dict(data.get("contract_snapshot"))
        else:
            contract_snapshot = _coerce_dict(current.get("contract_snapshot"))

        has_explicit_golden = "golden_output" in data
        if has_explicit_golden:
            golden_output = _json_safe(data.get("golden_output"))
        else:
            golden_output = _json_safe(current.get("golden_output")) if "golden_output" in current else None

        created_at = _as_text(current.get("created_at"), _now_iso())
        created_by_member_id = int(current.get("created_by_member_id") or actor_member_id or 0)
        run_history = _coerce_list(current.get("run_history"))
        if len(run_history) > MAX_RUN_HISTORY:
            run_history = run_history[:MAX_RUN_HISTORY]
        last_report = _coerce_dict(current.get("last_report"))

        normalized: dict[str, Any] = {
            "case_id": case_id,
            "tool_name": tool_name,
            "fixture_name": fixture_name,
            "description": description,
            "execution_mode": execution_mode,
            "input_args": _json_safe(input_args),
            "enabled": enabled,
            "tags": tags,
            "contract_snapshot": _json_safe(contract_snapshot),
            "golden_output": golden_output,
            "golden_hash": _json_hash(golden_output) if golden_output is not None else None,
            "last_report": _json_safe(last_report) if last_report else {},
            "run_history": _json_safe(run_history),
            "created_at": created_at,
            "created_by_member_id": created_by_member_id,
            "updated_at": _now_iso(),
            "updated_by_member_id": int(actor_member_id or 0),
        }
        return normalized

    def list_cases(self, *, include_disabled: bool = True, tool_name: str | None = None) -> list[dict[str, Any]]:
        filter_tool = _as_text(tool_name).lower()
        with self._lock:
            payload = self._load()
            output: list[dict[str, Any]] = []
            for value in payload.get("cases", {}).values():
                case = _coerce_dict(value)
                if not case:
                    continue
                if not include_disabled and not _as_bool(case.get("enabled"), True):
                    continue
                if filter_tool and _as_text(case.get("tool_name")).lower() != filter_tool:
                    continue
                output.append(_json_safe(case))
            output.sort(key=lambda item: _as_text(item.get("updated_at")), reverse=True)
            return output

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        clean_id = _as_text(case_id)
        if not clean_id:
            return None
        with self._lock:
            payload = self._load()
            case = _coerce_dict(payload.get("cases", {}).get(clean_id))
            if not case:
                return None
            return _json_safe(case)

    def upsert_case(
        self,
        payload: dict[str, Any] | None,
        *,
        actor_member_id: int,
    ) -> dict[str, Any]:
        data = _coerce_dict(payload)
        with self._lock:
            store = self._load()
            cases = store.get("cases")
            if not isinstance(cases, dict):
                cases = {}
                store["cases"] = cases
            case_id = _as_text(data.get("case_id"))
            existing = _coerce_dict(cases.get(case_id)) if case_id else {}
            normalized = self._normalize_case(data, existing=existing, actor_member_id=actor_member_id)
            cases[normalized["case_id"]] = normalized
            store["cases"] = cases
            self._save(store)
            return _json_safe(normalized)

    def remove_case(self, case_id: str) -> bool:
        clean_id = _as_text(case_id)
        if not clean_id:
            return False
        with self._lock:
            store = self._load()
            cases = store.get("cases")
            if not isinstance(cases, dict):
                return False
            if clean_id not in cases:
                return False
            cases.pop(clean_id, None)
            store["cases"] = cases
            self._save(store)
            return True

    def record_run(
        self,
        case_id: str,
        report: dict[str, Any],
        *,
        actor_member_id: int,
    ) -> dict[str, Any]:
        clean_id = _as_text(case_id)
        if not clean_id:
            raise ToolHarnessValidationError("case_id is required.")
        run_report = _json_safe(report)
        with self._lock:
            store = self._load()
            cases = store.get("cases")
            if not isinstance(cases, dict):
                raise ToolHarnessValidationError("Harness store is invalid.")
            existing = _coerce_dict(cases.get(clean_id))
            if not existing:
                raise ToolHarnessValidationError("Tool harness case not found.")

            history = _coerce_list(existing.get("run_history"))
            history.insert(0, run_report)
            history = history[:MAX_RUN_HISTORY]

            existing["last_report"] = run_report
            existing["run_history"] = _json_safe(history)
            existing["updated_at"] = _now_iso()
            existing["updated_by_member_id"] = int(actor_member_id or 0)
            cases[clean_id] = existing
            store["cases"] = cases
            self._save(store)
            return _json_safe(existing)

    def set_golden_output(
        self,
        case_id: str,
        golden_output: Any,
        *,
        actor_member_id: int,
    ) -> dict[str, Any]:
        clean_id = _as_text(case_id)
        if not clean_id:
            raise ToolHarnessValidationError("case_id is required.")
        with self._lock:
            store = self._load()
            cases = store.get("cases")
            if not isinstance(cases, dict):
                raise ToolHarnessValidationError("Harness store is invalid.")
            existing = _coerce_dict(cases.get(clean_id))
            if not existing:
                raise ToolHarnessValidationError("Tool harness case not found.")
            safe_output = _json_safe(golden_output)
            existing["golden_output"] = safe_output
            existing["golden_hash"] = _json_hash(safe_output)
            existing["updated_at"] = _now_iso()
            existing["updated_by_member_id"] = int(actor_member_id or 0)
            cases[clean_id] = existing
            store["cases"] = cases
            self._save(store)
            return _json_safe(existing)

    def set_contract_snapshot(
        self,
        case_id: str,
        contract_snapshot: dict[str, Any],
        *,
        actor_member_id: int,
    ) -> dict[str, Any]:
        clean_id = _as_text(case_id)
        if not clean_id:
            raise ToolHarnessValidationError("case_id is required.")
        with self._lock:
            store = self._load()
            cases = store.get("cases")
            if not isinstance(cases, dict):
                raise ToolHarnessValidationError("Harness store is invalid.")
            existing = _coerce_dict(cases.get(clean_id))
            if not existing:
                raise ToolHarnessValidationError("Tool harness case not found.")
            existing["contract_snapshot"] = _json_safe(_coerce_dict(contract_snapshot))
            existing["updated_at"] = _now_iso()
            existing["updated_by_member_id"] = int(actor_member_id or 0)
            cases[clean_id] = existing
            store["cases"] = cases
            self._save(store)
            return _json_safe(existing)
