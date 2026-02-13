# WO-DOC-024: Legacy Knowledge Migration, Cutover, and Operational Runbooks

Source item: User request (February 10, 2026) requiring a complete planning path from current flat knowledge storage to hierarchical document retrieval.

## 1. Verbose Description of Work Order
Plan and execute migration from the existing flat `knowledge` table to the new hierarchical document system with phased cutover and rollback safety.

This work order controls adoption risk and ensures continuity during transition.

Scope includes:
1. Legacy-to-new schema mapping and backfill plan.
2. Dual-write or shadow-index period.
3. Read-path cutover flags and rollback strategy.
4. Operator runbooks for migration, verification, and incident response.

## 2. Expression of Affected Files
Primary files:
1. New: `scripts/migrate_legacy_knowledge_to_documents.py`
2. New: `scripts/verify_document_migration.sql`
3. `hypermindlabs/utils.py` (`KnowledgeManager` migration and compatibility shims)
4. New: `docs/document-migration-runbook.md`

Secondary/API-surface files:
1. `docs/troubleshooting-startup.md` (migration checks)
2. `docs/prerequisites.md` (resource and rollout prerequisites)
3. `tests/test_legacy_document_migration.py`
4. `tests/test_document_cutover_flags.py`

## 3. Success and Validation Metrics
Valid outcome:
1. Legacy records migrate with preserved scope and provenance fields.
2. Dual-read validation confirms parity before cutover.
3. Cutover can be rolled back safely.
4. Runbooks are complete and tested in staging.

Partial outcome:
1. Migration script exists but parity validation is absent.
2. Cutover works only as one-way operation.
3. Legacy compatibility path remains undocumented.

Validation method:
1. Dry-run migration in staging with checksum comparison.
2. Shadow retrieval comparison old vs new paths.
3. Rollback rehearsal and post-rollback integrity checks.

## 4. Potential Failure Modes
1. Legacy records lack scope metadata needed for safe migration.
2. Dual-write introduces divergence between old and new stores.
3. Cutover toggles are not atomic across services.
4. Rollback leaves partial indexes and stale pointers.
