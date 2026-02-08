function trimText(value, maxLength = 140) {
    const text = String(value || "").trim();
    if (text.length <= maxLength) {
        return text;
    }
    return `${text.slice(0, maxLength - 3)}...`;
}

function statusClass(status) {
    const normalized = String(status || "info").toLowerCase();
    if (normalized === "error" || normalized === "failed") {
        return "error";
    }
    if (normalized === "completed" || normalized === "success") {
        return "success";
    }
    if (normalized === "cancelled") {
        return "cancelled";
    }
    return "info";
}

class TracePane {
    constructor(containerElement, onSelect = null) {
        this.container = containerElement;
        this.onSelect = onSelect;
        this.activeSeq = null;
        this.events = [];
        this.filters = {
            query: "",
            eventType: "all",
            status: "all",
        };
        this.render();
    }

    setOnSelect(callback) {
        this.onSelect = callback;
    }

    setEvents(events) {
        this.events = Array.isArray(events) ? events.slice() : [];
        this.render();
    }

    appendEvent(event) {
        if (!event || typeof event !== "object") {
            return;
        }
        const seq = Number.parseInt(event.seq, 10);
        const existingIndex = this.events.findIndex((item) => Number.parseInt(item.seq, 10) === seq);
        if (existingIndex >= 0) {
            this.events[existingIndex] = event;
        } else {
            this.events.push(event);
            this.events.sort((a, b) => Number.parseInt(a.seq, 10) - Number.parseInt(b.seq, 10));
        }
        this.render();
    }

    selectSeq(seq) {
        this.activeSeq = Number.parseInt(seq, 10);
        if (Number.isNaN(this.activeSeq)) {
            this.activeSeq = null;
        }
        this.render();
    }

    _summaryForEvent(event) {
        const payload = (event && typeof event.payload === "object") ? event.payload : {};
        const eventType = String(event?.event_type || "");

        if (eventType === "run.token") {
            return trimText(String(payload.chunk || "").replace(/\s+/g, " "), 100) || "Streaming token chunk";
        }
        if (payload.error) {
            return trimText(`Error: ${payload.error}`, 120);
        }
        if (payload.detail) {
            return trimText(payload.detail, 120);
        }
        if (payload.reason) {
            return trimText(`Reason: ${payload.reason}`, 120);
        }
        if (payload.model) {
            return `Model: ${payload.model}`;
        }
        if (payload.step_id) {
            return `Step: ${payload.step_id}`;
        }
        if (payload.item_index !== undefined) {
            return `Batch item: ${payload.item_index}`;
        }
        if (payload.response_preview) {
            return trimText(payload.response_preview, 120);
        }
        if (payload.stats && typeof payload.stats === "object") {
            const statKeys = Object.keys(payload.stats);
            return statKeys.length > 0 ? `Metrics: ${statKeys.slice(0, 4).join(", ")}` : "Metrics event";
        }
        return "No summary details available.";
    }

    _eventMatchesFilters(event) {
        const query = String(this.filters.query || "").trim().toLowerCase();
        const typeFilter = String(this.filters.eventType || "all");
        const statusFilter = String(this.filters.status || "all");

        if (typeFilter !== "all" && String(event.event_type) !== typeFilter) {
            return false;
        }
        if (statusFilter !== "all" && String(event.status) !== statusFilter) {
            return false;
        }
        if (!query) {
            return true;
        }

        const haystack = [
            event.seq,
            event.event_type,
            event.stage,
            event.status,
            this._summaryForEvent(event),
            JSON.stringify(event.payload || {}),
        ]
            .map((part) => String(part || "").toLowerCase())
            .join(" ");
        return haystack.includes(query);
    }

    _renderFilters(target) {
        const wrapper = document.createElement("div");
        wrapper.className = "ap-pane-toolbar";

        const queryInput = document.createElement("input");
        queryInput.type = "text";
        queryInput.placeholder = "Filter stage/event/payload";
        queryInput.value = this.filters.query;
        queryInput.className = "ap-pane-input";
        queryInput.addEventListener("input", () => {
            this.filters.query = queryInput.value;
            this.render();
        });
        wrapper.appendChild(queryInput);

        const eventTypeSelect = document.createElement("select");
        eventTypeSelect.className = "ap-pane-select";
        const eventTypes = ["all", ...new Set(this.events.map((event) => String(event.event_type || "")))];
        for (const eventType of eventTypes) {
            if (!eventType) {
                continue;
            }
            const option = document.createElement("option");
            option.value = eventType;
            option.textContent = eventType === "all" ? "All events" : eventType;
            if (eventType === this.filters.eventType) {
                option.selected = true;
            }
            eventTypeSelect.appendChild(option);
        }
        eventTypeSelect.addEventListener("change", () => {
            this.filters.eventType = eventTypeSelect.value;
            this.render();
        });
        wrapper.appendChild(eventTypeSelect);

        const statusSelect = document.createElement("select");
        statusSelect.className = "ap-pane-select";
        const statuses = ["all", ...new Set(this.events.map((event) => String(event.status || "")))];
        for (const status of statuses) {
            if (!status) {
                continue;
            }
            const option = document.createElement("option");
            option.value = status;
            option.textContent = status === "all" ? "All statuses" : status;
            if (status === this.filters.status) {
                option.selected = true;
            }
            statusSelect.appendChild(option);
        }
        statusSelect.addEventListener("change", () => {
            this.filters.status = statusSelect.value;
            this.render();
        });
        wrapper.appendChild(statusSelect);

        target.appendChild(wrapper);
    }

    _buildRow(event) {
        const row = document.createElement("button");
        row.className = "ap-trace-row";
        row.type = "button";
        row.dataset.seq = String(event.seq);

        if (this.activeSeq !== null && Number.parseInt(event.seq, 10) === this.activeSeq) {
            row.classList.add("active");
        }

        const headline = document.createElement("div");
        headline.className = "ap-trace-headline";

        const seqNode = document.createElement("span");
        seqNode.className = "ap-badge ap-badge-mono";
        seqNode.textContent = `#${event.seq}`;
        headline.appendChild(seqNode);

        const typeNode = document.createElement("span");
        typeNode.className = "ap-badge";
        typeNode.textContent = String(event.event_type || "event");
        headline.appendChild(typeNode);

        const statusNode = document.createElement("span");
        statusNode.className = `ap-badge ap-badge-${statusClass(event.status)}`;
        statusNode.textContent = String(event.status || "info");
        headline.appendChild(statusNode);
        row.appendChild(headline);

        const stageNode = document.createElement("div");
        stageNode.className = "ap-row-stage";
        stageNode.textContent = String(event.stage || "-");
        row.appendChild(stageNode);

        const summaryNode = document.createElement("div");
        summaryNode.className = "ap-row-summary";
        summaryNode.textContent = this._summaryForEvent(event);
        row.appendChild(summaryNode);

        const detail = document.createElement("div");
        detail.className = "ap-row-meta";
        detail.textContent = String(event.timestamp || "");
        row.appendChild(detail);

        row.addEventListener("click", () => {
            this.activeSeq = Number.parseInt(event.seq, 10);
            if (typeof this.onSelect === "function") {
                this.onSelect(event);
            }
            this.render();
        });

        return row;
    }

    render() {
        if (!this.container) {
            return;
        }

        this.container.innerHTML = "";
        this._renderFilters(this.container);

        const events = this.events.filter((event) => this._eventMatchesFilters(event));
        if (events.length === 0) {
            const empty = document.createElement("div");
            empty.className = "ap-empty";
            empty.textContent = this.events.length === 0
                ? "No run events yet. Start a run to capture traces."
                : "No events match current trace filters.";
            this.container.appendChild(empty);
            return;
        }

        for (const event of events) {
            this.container.appendChild(this._buildRow(event));
        }
    }
}

export { TracePane };
