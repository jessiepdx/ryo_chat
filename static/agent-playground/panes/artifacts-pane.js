function trimText(value, maxLength = 180) {
    const text = String(value || "").trim();
    if (text.length <= maxLength) {
        return text;
    }
    return `${text.slice(0, maxLength - 3)}...`;
}

function summarizeArtifact(artifactRecord) {
    const artifactType = String(artifactRecord?.artifact_type || "").toLowerCase();
    const payload = (artifactRecord && typeof artifactRecord.artifact === "object") ? artifactRecord.artifact : {};

    if (artifactType === "response") {
        return trimText(payload.text || payload.response || "Final response artifact.", 220);
    }
    if (artifactType === "workflow") {
        const steps = Array.isArray(payload.steps) ? payload.steps : [];
        const completed = steps.filter((item) => item && String(item.output || "").trim()).length;
        return `Workflow steps: ${steps.length} total, ${completed} with output.`;
    }
    if (artifactType === "batch") {
        const results = Array.isArray(payload.results) ? payload.results : [];
        const completed = results.filter((item) => item?.status === "completed").length;
        const failed = results.filter((item) => item?.status === "failed").length;
        return `Batch results: ${completed} completed, ${failed} failed, ${results.length} total.`;
    }
    if (artifactType === "compare") {
        const results = Array.isArray(payload.results) ? payload.results : [];
        const models = results.map((item) => item?.model).filter(Boolean);
        return `Model compare outputs for: ${models.join(", ") || "unknown models"}.`;
    }
    const keys = Object.keys(payload);
    if (keys.length > 0) {
        return `Keys: ${keys.slice(0, 6).join(", ")}${keys.length > 6 ? "..." : ""}`;
    }
    return "No artifact payload summary.";
}

class ArtifactsPane {
    constructor(containerElement) {
        this.container = containerElement;
        this.artifacts = [];
        this.render();
    }

    setArtifacts(artifacts) {
        this.artifacts = Array.isArray(artifacts) ? artifacts.slice() : [];
        this.render();
    }

    _appendRawJSON(target, payload, title) {
        const details = document.createElement("details");
        details.className = "ap-details";

        const summary = document.createElement("summary");
        summary.textContent = title;
        details.appendChild(summary);

        const body = document.createElement("pre");
        body.className = "ap-json";
        body.textContent = JSON.stringify(payload || {}, null, 2);
        details.appendChild(body);
        target.appendChild(details);
    }

    _appendWorkflowPreview(target, payload) {
        const steps = Array.isArray(payload.steps) ? payload.steps : [];
        if (steps.length === 0) {
            return;
        }

        const list = document.createElement("ul");
        list.className = "ap-list";
        for (const step of steps.slice(0, 5)) {
            const item = document.createElement("li");
            const stepID = String(step?.step_id || step?.step_index || "step");
            item.textContent = `${stepID}: ${trimText(step?.output || "(no output)", 120)}`;
            list.appendChild(item);
        }
        target.appendChild(list);
    }

    _appendBatchPreview(target, payload) {
        const results = Array.isArray(payload.results) ? payload.results : [];
        if (results.length === 0) {
            return;
        }

        const list = document.createElement("ul");
        list.className = "ap-list";
        for (const result of results.slice(0, 6)) {
            const item = document.createElement("li");
            const index = result?.item_index || "?";
            const status = String(result?.status || "unknown");
            const response = result?.response || result?.error || "(empty)";
            item.textContent = `#${index} [${status}] ${trimText(response, 120)}`;
            list.appendChild(item);
        }
        target.appendChild(list);
    }

    _appendComparePreview(target, payload) {
        const results = Array.isArray(payload.results) ? payload.results : [];
        if (results.length === 0) {
            return;
        }

        const list = document.createElement("ul");
        list.className = "ap-list";
        for (const result of results.slice(0, 6)) {
            const item = document.createElement("li");
            const model = String(result?.model || "model");
            const status = String(result?.status || "unknown");
            const response = result?.response || result?.error || "(empty)";
            item.textContent = `${model} [${status}] ${trimText(response, 120)}`;
            list.appendChild(item);
        }
        target.appendChild(list);
    }

    _buildArtifactCard(artifactRecord) {
        const payload = (artifactRecord && typeof artifactRecord.artifact === "object") ? artifactRecord.artifact : {};
        const card = document.createElement("div");
        card.className = "ap-card";

        const heading = document.createElement("div");
        heading.className = "ap-card-headline";
        heading.textContent = `${artifactRecord.artifact_type} / ${artifactRecord.artifact_name}`;
        card.appendChild(heading);

        const meta = document.createElement("div");
        meta.className = "ap-row-meta";
        const stepSeq = artifactRecord.step_seq ? `step ${artifactRecord.step_seq} | ` : "";
        meta.textContent = `${stepSeq}${artifactRecord.timestamp || ""}`;
        card.appendChild(meta);

        const summary = document.createElement("div");
        summary.className = "ap-row-summary";
        summary.textContent = summarizeArtifact(artifactRecord);
        card.appendChild(summary);

        const artifactType = String(artifactRecord?.artifact_type || "").toLowerCase();
        if (artifactType === "workflow") {
            this._appendWorkflowPreview(card, payload);
        } else if (artifactType === "batch") {
            this._appendBatchPreview(card, payload);
        } else if (artifactType === "compare") {
            this._appendComparePreview(card, payload);
        } else if (artifactType === "response" && payload.text) {
            const responseNode = document.createElement("pre");
            responseNode.className = "ap-json";
            responseNode.textContent = String(payload.text);
            card.appendChild(responseNode);
        }

        this._appendRawJSON(card, payload, "Raw Artifact JSON");
        return card;
    }

    render() {
        if (!this.container) {
            return;
        }

        this.container.innerHTML = "";
        if (this.artifacts.length === 0) {
            const empty = document.createElement("div");
            empty.className = "ap-empty";
            empty.textContent = "No artifacts yet. Responses, workflow outputs, and compare snapshots will appear here.";
            this.container.appendChild(empty);
            return;
        }

        for (const artifact of this.artifacts.slice().reverse()) {
            this.container.appendChild(this._buildArtifactCard(artifact));
        }
    }
}

export { ArtifactsPane };
