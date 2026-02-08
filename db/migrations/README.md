# Database Migrations

This folder contains SQL migrations used by runtime bootstrap managers and setup scripts.

## Conventions

- One migration step per file.
- Files are ordered by numeric prefix.
- Runtime code executes these files idempotently (`CREATE TABLE IF NOT EXISTS`, safe `ALTER` where possible).
- Template tokens use `{{token_name}}` and are rendered by code before execution.
  - Current token usage: `{{vector_dimensions}}`.

## Current Coverage

The runtime managers in `hypermindlabs/utils.py` load migrations from this folder for:

- member tables
- chat history tables
- community tables
- community score table
- knowledge and retrieval tables
- proposal tables
- spam table
- inference usage table
- pgvector extension enablement
