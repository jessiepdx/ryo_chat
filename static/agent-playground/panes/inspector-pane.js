function createKeyValueTable(values) {
    const table = document.createElement("div");
    table.className = "ap-kv-grid";

    for (const [key, value] of Object.entries(values || {})) {
        const keyNode = document.createElement("div");
        keyNode.className = "ap-kv-key";
        keyNode.textContent = key;
        table.appendChild(keyNode);

        const valueNode = document.createElement("div");
        valueNode.className = "ap-kv-value";
        valueNode.textContent = typeof value === "object" ? JSON.stringify(value) : String(value);
        table.appendChild(valueNode);
    }
    return table;
}

class InspectorPane {
    constructor(containerElement) {
        this.container = containerElement;
        this.event = null;
        this.metrics = null;
        this.schemaValidation = null;
        this.render();
    }

    setEvent(event) {
        this.event = event;
        this.render();
    }

    setMetrics(metrics) {
        this.metrics = metrics;
        this.render();
    }

    setSchemaValidation(validation) {
        this.schemaValidation = validation;
        this.render();
    }

    _appendCard(title, className = "") {
        const card = document.createElement("section");
        card.className = `ap-card ${className}`.trim();
        const heading = document.createElement("div");
        heading.className = "ap-card-headline";
        heading.textContent = title;
        card.appendChild(heading);
        this.container.appendChild(card);
        return card;
    }

    _renderEventInspector() {
        if (!this.event) {
            return;
        }

        const payload = (this.event && typeof this.event.payload === "object") ? this.event.payload : {};
        const eventCard = this._appendCard("Selected Event");

        const header = document.createElement("div");
        header.className = "ap-inspector-header";
        const seqNode = document.createElement("span");
        seqNode.className = "ap-badge ap-badge-mono";
        seqNode.textContent = `#${this.event.seq}`;
        header.appendChild(seqNode);

        const eventTypeNode = document.createElement("span");
        eventTypeNode.className = "ap-badge";
        eventTypeNode.textContent = String(this.event.event_type || "event");
        header.appendChild(eventTypeNode);

        const statusNode = document.createElement("span");
        statusNode.className = "ap-badge";
        statusNode.textContent = String(this.event.status || "info");
        header.appendChild(statusNode);
        eventCard.appendChild(header);

        const meta = document.createElement("div");
        meta.className = "ap-row-meta";
        meta.textContent = `${this.event.stage || "-"} | ${this.event.timestamp || ""}`;
        eventCard.appendChild(meta);

        const summaryPayload = {};
        for (const key of ["detail", "reason", "error", "model", "step_id", "item_index", "response_preview"]) {
            if (payload[key] !== undefined) {
                summaryPayload[key] = payload[key];
            }
        }
        if (Object.keys(summaryPayload).length > 0) {
            eventCard.appendChild(createKeyValueTable(summaryPayload));
        }

        if (payload.stats && typeof payload.stats === "object") {
            const statsCard = this._appendCard("Event Metrics");
            statsCard.appendChild(createKeyValueTable(payload.stats));
        }

        const rawPayload = document.createElement("details");
        rawPayload.className = "ap-details";
        rawPayload.open = false;
        const summary = document.createElement("summary");
        summary.textContent = "Raw Event JSON";
        rawPayload.appendChild(summary);
        const body = document.createElement("pre");
        body.className = "ap-json";
        body.textContent = JSON.stringify(this.event, null, 2);
        rawPayload.appendChild(body);
        eventCard.appendChild(rawPayload);
    }

    _renderPanelMetrics() {
        if (!this.metrics || typeof this.metrics !== "object") {
            return;
        }
        const metricsCard = this._appendCard("Run Panel Metrics");
        metricsCard.appendChild(createKeyValueTable(this.metrics));
    }

    _renderSchemaValidation() {
        if (!this.schemaValidation) {
            return;
        }
        const schemaCard = this._appendCard("Schema Validation");
        const valid = Boolean(this.schemaValidation.valid);

        const headline = document.createElement("div");
        headline.className = `ap-row-summary ${valid ? "ap-ok" : "ap-error"}`;
        headline.textContent = valid
            ? "Current schema options are valid."
            : "Schema options have validation errors. Fix before starting run.";
        schemaCard.appendChild(headline);

        if (Array.isArray(this.schemaValidation.errors) && this.schemaValidation.errors.length > 0) {
            const errorList = document.createElement("ul");
            errorList.className = "ap-list";
            for (const error of this.schemaValidation.errors) {
                const item = document.createElement("li");
                item.textContent = String(error);
                errorList.appendChild(item);
            }
            schemaCard.appendChild(errorList);
        }

        const valueJSON = document.createElement("details");
        valueJSON.className = "ap-details";
        const summary = document.createElement("summary");
        summary.textContent = "Resolved Schema Values";
        valueJSON.appendChild(summary);
        const body = document.createElement("pre");
        body.className = "ap-json";
        body.textContent = JSON.stringify(this.schemaValidation.value || {}, null, 2);
        valueJSON.appendChild(body);
        schemaCard.appendChild(valueJSON);
    }

    render() {
        if (!this.container) {
            return;
        }

        this.container.innerHTML = "";
        if (!this.event && !this.metrics && !this.schemaValidation) {
            const empty = document.createElement("div");
            empty.className = "ap-empty";
            empty.textContent = "Select a trace event to inspect payloads and step metadata.";
            this.container.appendChild(empty);
            return;
        }

        this._renderPanelMetrics();
        this._renderSchemaValidation();
        this._renderEventInspector();
    }
}

export { InspectorPane };
