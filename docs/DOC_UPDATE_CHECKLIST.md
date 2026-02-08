# Documentation Update Checklist

Use this checklist for every change that affects setup, config/env schemas, runtime behavior, fallbacks, policies, or operator workflows.

## 1. Trigger Check
1. Confirm whether code changes affect any of:
   - `config.json` keys
   - `.env` keys
   - startup behavior
   - fallback behavior
   - policy files
   - scripts under `scripts/`
2. If no documentation-facing behavior changed, note rationale in PR.

## 2. Required File Updates (Order Matters)
1. Update `docs/master-engineering.md`:
   - status snapshot
   - architecture/plan changes
2. Update `readme.md`:
   - setup commands
   - prerequisites
   - key behavior notes
3. Update templates:
   - `config.empty.json`
   - `.env.example`
4. Update this changelog:
   - `docs/CHANGELOG_ENGINEERING.md`

## 3. Setup and Endpoint Consistency Checks
1. Confirm Ollama endpoint precedence is documented consistently:
   - explicit custom host
   - existing configured host
   - default local `http://127.0.0.1:11434`
2. Confirm DB routing behavior is documented consistently:
   - primary route
   - fallback route
   - degraded/failed status handling
3. Confirm setup scripts preserve unrelated config sections when editing one domain.

## 4. Policy and Secondary File Checks
1. If policy behavior changed, update:
   - `policies/agent/*.json` documentation references
   - `policies/agent/system_prompt/*.txt` usage notes
2. If PR process changes, update:
   - `.github/PULL_REQUEST_TEMPLATE.md`

## 5. Validation Before Merge
1. Run `python3 -m compileall -q .` if Python files changed.
2. Run relevant tests for changed behavior.
3. Verify all doc links and file paths referenced in docs exist.
4. Confirm no contradictory guidance exists between README and engineering docs.

## 6. Completion Record
1. Add changelog entry in `docs/CHANGELOG_ENGINEERING.md`.
2. Mark checklist completion in PR description.
