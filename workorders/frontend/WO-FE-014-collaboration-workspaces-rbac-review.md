# WO-FE-014: Collaboration, Workspaces, RBAC, and Review Workflows

Source item: `docs/master-engineering.md` section 4.7, and TODO-FE-029 through TODO-FE-030.

## 1. Verbose Description of Work Order
Implement multi-user collaboration primitives so teams can share runs/configurations safely across projects with role-scoped permissions.

Scope:
1. Add workspace/project hierarchy and membership model.
2. Enforce RBAC across routes and APIs (viewer/editor/admin).
3. Add shareable run permalinks with access checks.
4. Add threaded comments/review notes on trace steps and configuration revisions.
5. Add approval workflows for sensitive configuration changes (tool/policy changes).

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/workspaces.py` (workspace + membership service)
2. New: `hypermindlabs/rbac.py` (role policy checks)
3. `web_ui.py` (workspace routing + enforcement + share links)
4. `templates/base-html.html` (workspace selector shell)
5. New: `templates/review-center.html`
6. New: `static/agent-playground/collab/workspaces.js`
7. New: `static/agent-playground/collab/reviews.js`

Secondary/API-surface files:
1. `hypermindlabs/utils.py` (member metadata helpers)
2. `hypermindlabs/run_manager.py` (attach workspace/project IDs to runs)
3. New: `tests/test_rbac.py`
4. New: `tests/test_workspace_access.py`
5. `docs/master-engineering.md` (status updates)

Data and schema surfaces:
1. Workspace, project, membership, and role-mapping tables.
2. Comment/review thread tables linked to run IDs and step IDs.
3. Signed permalink token model with expiration policy.

## 3. Success and Validation Metrics
Valid outcome:
1. Workspace/project scoping is enforced for all run/config objects.
2. RBAC prevents unauthorized edits while preserving read-only views.
3. Users can share run traces via permalinks within permission boundaries.
4. Review comments are persisted, ordered, and auditable.

Partial outcome:
1. Workspace model exists but endpoints still leak cross-workspace data.
2. Shared links work but bypass role checks.
3. Review comments exist without step-level linkage.

Validation method:
1. Access-control integration tests for each role and endpoint class.
2. Negative tests verifying cross-workspace isolation.
3. UI tests for review comment creation, edit restrictions, and visibility.

## 4. Potential Failure Modes
1. Missing RBAC checks on non-obvious APIs (for example export/download routes).
2. Permalink tokens are guessable or long-lived beyond policy.
3. Workspace IDs not attached at run creation, causing orphaned records.
4. Comment threads leak sensitive payload snippets without redaction.
