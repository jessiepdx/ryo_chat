function buildBatchPayload(inputText, optionValues = {}) {
    const batchInputs = String(inputText || "")
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line.length > 0);

    return {
        mode: "batch",
        message: String(inputText || "").trim(),
        batch_inputs: batchInputs,
        options: optionValues,
    };
}

export { buildBatchPayload };
