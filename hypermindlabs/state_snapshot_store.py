from __future__ import annotations

from copy import deepcopy
from typing import Any


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    return {}


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def select_snapshot_for_seq(
    snapshots: list[dict[str, Any]] | None,
    step_seq: int | None = None,
) -> dict[str, Any] | None:
    if not isinstance(snapshots, list) or not snapshots:
        return None

    normalized: list[tuple[int, dict[str, Any]]] = []
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        seq = _safe_int(snapshot.get("step_seq"))
        if seq is None:
            continue
        normalized.append((seq, snapshot))

    if not normalized:
        return None

    normalized.sort(key=lambda item: item[0])

    if step_seq is None:
        return deepcopy(normalized[-1][1])

    target = _safe_int(step_seq)
    if target is None:
        return deepcopy(normalized[-1][1])

    selected: dict[str, Any] | None = None
    for seq, snapshot in normalized:
        if seq <= target:
            selected = snapshot
        else:
            break

    if selected is not None:
        return deepcopy(selected)

    return deepcopy(normalized[0][1])


def build_replay_state_plan(
    snapshots: list[dict[str, Any]] | None,
    replay_from_seq: int | None,
    state_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    selected = select_snapshot_for_seq(snapshots, replay_from_seq)
    base_state = _coerce_dict(selected.get("state")) if isinstance(selected, dict) else {}
    overrides = _coerce_dict(state_overrides)
    merged_state = _deep_merge(base_state, overrides)

    selected_seq = _safe_int(selected.get("step_seq")) if isinstance(selected, dict) else None

    return {
        "selected_snapshot_seq": selected_seq,
        "base_state": base_state,
        "state_overrides": overrides,
        "merged_state": merged_state,
        "override_keys": sorted(overrides.keys()),
    }
