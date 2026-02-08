function _inferWorkflowSteps(inputText) {
    const lines = String(inputText || "")
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line.length > 0);

    if (lines.length <= 1) {
        return [];
    }

    return lines.map((line, index) => ({
        id: `step_${index + 1}`,
        prompt: line,
    }));
}

function buildWorkflowPayload(inputText, optionValues = {}) {
    return {
        mode: "workflow",
        message: String(inputText || "").trim(),
        workflow_steps: _inferWorkflowSteps(inputText),
        options: optionValues,
    };
}

export { buildWorkflowPayload };
