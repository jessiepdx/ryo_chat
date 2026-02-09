# WO-007: Tool-Calling Stack Review and Improvements

Source item: `docs/master-engineering.md` section 2 (planned upgrade #7), section 3.2, section 4.5, section 6 phase 4.

## 1. Verbose Description of Work Order
Perform a complete hardening pass of the tool-calling stack from prompt/schema definition through execution and result propagation.

This work order should eliminate ambiguity in function argument handling, unify tool metadata definitions, and ensure consistent behavior whether model output matches schema perfectly or not. It should also close existing gaps where tool arguments may be misread or where unavailable tool names are silently ignored without structured fallback.

Core improvements:
1. Canonical tool registry with explicit schemas.
2. Strict argument extraction/parsing and coercion rules.
3. Controlled handling for unknown tool names.
4. Standard tool result envelope.
5. Integration with `ToolRuntime` and fallback behavior.

## 2. Expression of Affected Files
Primary files:
1. `hypermindlabs/agents.py` (tool definitions, tool dispatch path, `available_functions`, `toolCaller`)
2. New: `hypermindlabs/tool_registry.py`
3. New: `hypermindlabs/tool_runtime.py` (if not already created in WO-004)
4. New: `tests/test_tool_calling_agent.py`

Secondary/API-surface files:
1. `policies/agent/system_prompt/tool_calling_sp.txt` (align instruction language with strict schema)
2. `policies/agent/tool_calling_policy.json` (tool usage behavior flags)
3. `readme.md` (tool support and limitation notes)
4. `docs/master-engineering.md` (status and decisions)

Runtime variables and configuration surfaces:
1. `api_keys.brave_search`
2. `knowledge.domains` (tool description context)
3. Model policy for tool-calling agent

## 3. Success and Validation Metrics
Valid outcome:
1. Tool arguments parsed consistently across all supported tools.
2. Unknown tool invocation handled with structured fallback result.
3. Tool results use a stable schema (`tool_name`, `status`, `tool_results`, `error` as needed).
4. Tool-calling tests cover valid, invalid, missing-arg, and unknown-tool cases.

Partial outcome:
1. Some tools migrated to registry while others remain ad hoc.
2. Unknown tools no longer crash, but diagnostics are weak.
3. Argument validation present but bypassed in some call paths.

Validation method:
1. Use mock tool-call outputs with malformed arguments.
2. Verify orchestration continues and logs actionable diagnostics.
3. Run integration flow through `ConversationOrchestrator`.

## 4. Potential Failure Modes
1. Overly strict parser rejects model outputs that were previously usable.
2. Schema mismatch between tool prompt instructions and runtime validation.
3. Tool registry drift from real function signatures.
4. Backward compatibility break for existing serialized tool-response expectations.
