function buildChatPayload(inputText, optionValues = {}) {
    return {
        mode: "chat",
        message: String(inputText || "").trim(),
        options: optionValues,
    };
}

export { buildChatPayload };
