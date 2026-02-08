# Policy Guide

This guide covers policy validation and guided editing for:
- `policies/agent/*_policy.json`
- `policies/agent/system_prompt/*_sp.txt`

## What Gets Validated
1. Policy file exists and contains valid JSON.
2. Required policy keys:
   - `allow_custom_system_prompt` (boolean)
   - `allowed_models` (non-empty list of model names)
3. System prompt file exists and is readable.
4. Policy model compatibility against a live Ollama model inventory.

Model compatibility can run in:
- warning mode (default): mismatches are warnings.
- strict mode: mismatches are treated as validation errors.

## Endpoint Precedence
Policy model discovery uses deterministic Ollama host precedence:
1. Explicit override (`--ollama-host`)
2. Host inferred from `config.json` `inference.*.url`
3. Default local host: `http://127.0.0.1:11434`

## Run Validation
Validate one policy:
```bash
python3 scripts/policy_wizard.py --policy tool_calling --validate-only
```

Validate with strict model checks:
```bash
python3 scripts/policy_wizard.py --policy tool_calling --validate-only --strict-models
```

## Guided Policy Editing
Interactive walkthrough:
```bash
python3 scripts/policy_wizard.py
```

The walkthrough lets you:
1. Choose a policy.
2. Preview the linked system prompt.
3. Toggle `allow_custom_system_prompt`.
4. Select `allowed_models` from discovered models (or manual names).
5. Save with backup + post-save validation.

If post-save validation fails, the wizard rolls back to the previous policy file.

## Non-Interactive Updates
Example:
```bash
python3 scripts/policy_wizard.py \
  --policy chat_conversation \
  --non-interactive \
  --allow-custom-system-prompt false \
  --allowed-models llama3.2:latest,gemma3:4b
```
