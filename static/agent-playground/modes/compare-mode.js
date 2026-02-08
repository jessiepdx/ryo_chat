function _coerceModelList(optionValues) {
    const explicit = optionValues.compare_models;
    if (Array.isArray(explicit)) {
        return explicit.map((item) => String(item).trim()).filter((item) => item.length > 0);
    }

    if (typeof explicit === "string") {
        return explicit
            .split(",")
            .map((item) => item.trim())
            .filter((item) => item.length > 0);
    }

    const fallback = optionValues.model_requested;
    if (typeof fallback === "string" && fallback.trim()) {
        const model = fallback.trim();
        return [model, model];
    }

    return [];
}

function buildComparePayload(inputText, optionValues = {}) {
    const compareModels = _coerceModelList(optionValues);
    return {
        mode: "compare",
        message: String(inputText || "").trim(),
        compare_models: compareModels,
        options: optionValues,
    };
}

export { buildComparePayload };
