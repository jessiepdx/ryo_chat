# WO-FE-009: Memory and Context Strategy Controls

Source item: `docs/master-engineering.md` section 4.4 and TODO-FE-017.

## 1. Verbose Description of Work Order
Implement configurable memory/context strategies as first-class run settings in the playground.

Scope:
1. Expose short-term context controls:
- trimming strategy
- compression/summarization strategy
- token budget windows
2. Expose long-term memory pointers:
- episodic
- semantic
- procedural
3. Capture memory-write provenance metadata:
- author step
- confidence
- TTL/expiry
- evidence links
4. Add memory inspection UX per run and per agent definition.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/memory_manager.py`
2. `hypermindlabs/agents.py` (memory policy hooks + write events)
3. `web_ui.py` (memory strategy config endpoints)
4. New: `static/agent-playground/memory/memory-panel.js`
5. New: `static/agent-playground/memory/memory-policy-editor.js`

Secondary/API-surface files:
1. `hypermindlabs/run_events.py` (memory write events)
2. `hypermindlabs/runtime_settings.py` (memory strategy defaults)
3. New: `tests/test_memory_manager.py`
4. New: `tests/test_memory_policy_api.py`

## 3. Success and Validation Metrics
Valid outcome:
1. Memory strategy can be configured and applied per run or agent profile.
2. Memory writes are observable with provenance metadata.
3. Short-term trimming/compression visibly affects run context size and behavior.
4. Memory inspection UI can filter by type (episodic/semantic/procedural).

Partial outcome:
1. Strategy fields render in UI but do not alter runtime behavior.
2. Memory writes occur but provenance metadata is incomplete.
3. Only one memory class is implemented.

Validation method:
1. Integration tests with deterministic prompt sets and expected context-size changes.
2. Event validation tests for memory write metadata completeness.
3. UI tests for filtering and provenance display.

## 4. Potential Failure Modes
1. Over-aggressive trimming removes critical context and degrades output quality.
2. Compression summaries introduce hallucinated details.
3. Memory retention without TTL causes stale context contamination.
4. Provenance links break due to missing step identifiers.
