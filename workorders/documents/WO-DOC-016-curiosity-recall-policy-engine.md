# WO-DOC-016: Curiosity-Driven Recall Policy Engine

Source item: User request (February 10, 2026) requiring recall behavior that is driven by curiosity.

## 1. Verbose Description of Work Order
Implement a policy engine that decides when to trigger retrieval based on uncertainty, novelty, unresolved references, and evidence gaps.

This work order operationalizes curiosity as explicit, testable policy signals instead of ad hoc heuristics.

Scope includes:
1. Curiosity signal model (`uncertainty`, `novel_entity`, `missing_evidence`, `cross-turn_reference`).
2. Retrieval-trigger policy and budget controls by mode/scope.
3. Progressive deepening rounds with stop conditions.
4. Structured logs for why retrieval was or was not triggered.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/curiosity_policy.py`
2. New: `hypermindlabs/curiosity_signals.py`
3. `hypermindlabs/agents.py` (policy hook before retrieval)
4. `hypermindlabs/history_recall.py` (progressive deepening reuse/extension)

Secondary/API-surface files:
1. `hypermindlabs/runtime_settings.py` (curiosity thresholds and budgets)
2. `web_ui.py` (curiosity decision debug endpoint)
3. `tests/test_curiosity_policy.py`
4. `tests/test_curiosity_triggering.py`

## 3. Success and Validation Metrics
Valid outcome:
1. Retrieval trigger decisions are deterministic and explainable.
2. Curiosity policy improves factual grounding without excessive retrieval calls.
3. Progressive rounds respect latency and token budgets.
4. Policy decisions are visible in run/retrieval traces.

Partial outcome:
1. Curiosity signals exist but do not influence retrieval routing.
2. Retrieval is triggered but without reason codes.
3. Progressive rounds run without budget stop conditions.

Validation method:
1. Scenario tests for high/medium/low curiosity prompts.
2. Cost-quality analysis of retrieval call volume vs answer quality.
3. Regression tests for trigger stability.

## 4. Potential Failure Modes
1. Curiosity thresholds are too aggressive and over-trigger retrieval.
2. Signals are under-sensitive and miss evidence-needed queries.
3. Policy conflicts with explicit user instructions to avoid retrieval.
4. Trigger decisions are not persisted, reducing auditability.
