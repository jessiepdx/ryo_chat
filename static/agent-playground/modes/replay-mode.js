function buildReplayPayload(inputText, optionValues = {}) {
    const stepSeqRaw = optionValues.replay_from_seq ?? optionValues.step_seq;
    const stepSeq = Number.parseInt(stepSeqRaw, 10);

    return {
        mode: "replay",
        message: String(inputText || "").trim(),
        source_run_id: String(optionValues.source_run_id || "").trim(),
        replay_from_seq: Number.isNaN(stepSeq) ? undefined : stepSeq,
        state_overrides: (optionValues.state_overrides && typeof optionValues.state_overrides === "object")
            ? optionValues.state_overrides
            : {},
        options: optionValues,
    };
}

export { buildReplayPayload };
