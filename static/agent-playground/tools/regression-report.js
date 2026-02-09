function asText(value, fallback = "") {
    const text = String(value ?? "").trim();
    return text || fallback;
}

function asObject(value) {
    return value && typeof value === "object" ? value : {};
}

function toJSON(value) {
    return JSON.stringify(value ?? {}, null, 2);
}

function formatRegression(regression) {
    const payload = asObject(regression);
    const status = asText(payload.status, "unknown");
    const diffCount = Number.parseInt(payload.diff_count, 10) || 0;
    const message = asText(payload.message, "");
    const lines = [
        `Regression: ${status}${message ? ` (${message})` : ""}`,
        `Diffs: ${diffCount}`,
    ];
    if (diffCount > 0 && Array.isArray(payload.diffs)) {
        const sample = payload.diffs.slice(0, 10);
        lines.push("Diff sample:");
        for (const item of sample) {
            const row = asObject(item);
            lines.push(
                `- ${asText(row.path, "$")} [${asText(row.kind, "change")}] expected=${asText(row.expected, "-")} actual=${asText(row.actual, "-")}`,
            );
        }
        if (payload.diffs.length > sample.length) {
            lines.push(`... ${payload.diffs.length - sample.length} additional diff entries`);
        }
    }
    return lines.join("\n");
}

function formatContract(contract) {
    const payload = asObject(contract);
    const status = asText(payload.status, "unknown");
    const message = asText(payload.message, "");
    const lines = [
        `Contract: ${status}${message ? ` (${message})` : ""}`,
    ];
    const addedRequired = Array.isArray(payload.added_required) ? payload.added_required : [];
    const removedRequired = Array.isArray(payload.removed_required) ? payload.removed_required : [];
    const removedProps = Array.isArray(payload.removed_properties) ? payload.removed_properties : [];
    const typeChanges = Array.isArray(payload.type_changes) ? payload.type_changes : [];
    if (addedRequired.length > 0) {
        lines.push(`Added required: ${addedRequired.join(", ")}`);
    }
    if (removedRequired.length > 0) {
        lines.push(`Removed required: ${removedRequired.join(", ")}`);
    }
    if (removedProps.length > 0) {
        lines.push(`Removed properties: ${removedProps.join(", ")}`);
    }
    if (typeChanges.length > 0) {
        const sample = typeChanges.slice(0, 5);
        lines.push("Type changes:");
        for (const row of sample) {
            const item = asObject(row);
            const fromTypes = Array.isArray(item.expected) ? item.expected.join("|") : asText(item.expected, "-");
            const toTypes = Array.isArray(item.actual) ? item.actual.join("|") : asText(item.actual, "-");
            lines.push(`- ${asText(item.property, "?")}: ${fromTypes} -> ${toTypes}`);
        }
    }
    return lines.join("\n");
}

function formatHarnessReport(report) {
    const payload = asObject(report);
    if (!payload || Object.keys(payload).length === 0) {
        return "No harness report yet.";
    }

    const result = asObject(payload.result);
    const error = asObject(result.error);
    const lines = [
        `Case: ${asText(payload.case_id, "-")} (${asText(payload.tool_name, "-")})`,
        `Run At: ${asText(payload.run_at, "-")}`,
        `Mode: ${asText(payload.execution_mode, "-")} | Duration: ${asText(payload.duration_ms, "-")} ms`,
        `Result: ${asText(result.status, "unknown")} | Attempts: ${asText(result.attempts, "-")}`,
    ];
    if (error && Object.keys(error).length > 0) {
        lines.push(`Error: ${asText(error.code, "unknown")} - ${asText(error.message, "")}`);
    }
    lines.push("");
    lines.push(formatContract(payload.contract));
    lines.push("");
    lines.push(formatRegression(payload.regression));
    lines.push("");
    lines.push("Output:");
    lines.push(toJSON(result.tool_results));
    return lines.join("\n");
}

export { formatHarnessReport };
