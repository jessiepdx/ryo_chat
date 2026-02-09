async function cancelRun(apiBase, runId) {
    const response = await fetch(`${apiBase}/runs/${encodeURIComponent(runId)}/cancel`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
    });
    return response.json();
}

async function replayRun(apiBase, runId, payload = {}) {
    const response = await fetch(`${apiBase}/runs/${encodeURIComponent(runId)}/replay`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(payload || {}),
    });
    return response.json();
}

async function resumeRun(apiBase, runId) {
    const response = await fetch(`${apiBase}/runs/${encodeURIComponent(runId)}/resume`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
    });
    return response.json();
}

export { cancelRun, replayRun, resumeRun };
