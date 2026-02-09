from __future__ import annotations

from typing import Any

from hypermindlabs.run_manager import RunManager


class ReplayManager:
    """Convenience wrapper around RunManager replay operations."""

    def __init__(self, run_manager: RunManager):
        self._run_manager = run_manager

    def replay_from_start(self, run_id: str, *, auto_start: bool = True) -> dict[str, Any]:
        return self._run_manager.replay_run(run_id, auto_start=auto_start)

    def replay_from_step(self, run_id: str, step_seq: int, *, auto_start: bool = True) -> dict[str, Any]:
        return self._run_manager.replay_run(
            run_id,
            replay_from_seq=int(step_seq),
            auto_start=auto_start,
        )

    def replay_with_state(
        self,
        run_id: str,
        *,
        step_seq: int | None = None,
        state_overrides: dict[str, Any] | None = None,
        auto_start: bool = True,
    ) -> dict[str, Any]:
        return self._run_manager.replay_run(
            run_id,
            replay_from_seq=step_seq,
            state_overrides=state_overrides,
            auto_start=auto_start,
        )
