# WO-FE-015: Security, Governance, Audit, and Redaction Controls

Source item: `docs/master-engineering.md` section 4.7, and TODO-FE-031 through TODO-FE-033.

## 1. Verbose Description of Work Order
Implement governance controls required for safe multi-user operation: secret boundaries, audit trails, and redaction workflows.

Scope:
1. Enforce secret handling boundaries:
- never return plaintext secrets to frontend
- redact secrets in logs/events/artifacts
- support secret rotation metadata
2. Add audit event logging for:
- config changes
- policy changes
- tool permission changes
- privileged run actions
3. Add PII detection/redaction hooks on exported traces and artifacts.
4. Add governance dashboard showing high-risk events and actor provenance.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/security.py` (secret and redaction helpers)
2. New: `hypermindlabs/audit_log.py` (immutable audit event writer/reader)
3. `web_ui.py` (governance endpoints + redacted payload responses)
4. `hypermindlabs/tool_runtime.py` (sensitive-field masking in runtime events)
5. New: `templates/security-governance.html`
6. New: `static/agent-playground/security/governance.js`

Secondary/API-surface files:
1. `hypermindlabs/run_events.py` (classification flags for sensitive fields)
2. `hypermindlabs/policy_manager.py` (policy change audit emit)
3. New: `tests/test_security_redaction.py`
4. New: `tests/test_audit_log.py`
5. `readme.md` (secret handling expectations)

Data and schema surfaces:
1. Append-only audit log table with actor, action, resource, and timestamp indexes.
2. Redaction metadata model for reversible policy-approved masking.
3. Export response schema that includes redaction manifest.

## 3. Success and Validation Metrics
Valid outcome:
1. Secret values are never exposed in API responses, logs, or trace exports.
2. All sensitive change operations produce immutable audit records.
3. PII redaction hooks execute before trace/artifact export.
4. Governance dashboard can filter audit events by actor/resource/action/time.

Partial outcome:
1. Audit logs exist but are missing critical mutation paths.
2. Redaction covers logs but not artifact payloads.
3. Security UI exists without policy-bound enforcement.

Validation method:
1. Negative tests asserting secret fields are masked in all payload classes.
2. Audit completeness tests over privileged endpoints.
3. Export tests confirming redaction manifest integrity.

## 4. Potential Failure Modes
1. Secret leakage via nested JSON fields not covered by mask rules.
2. Audit write failures silently drop high-risk events.
3. Over-redaction removes information needed for debugging/compliance.
4. PII detection false positives degrade developer usability.
