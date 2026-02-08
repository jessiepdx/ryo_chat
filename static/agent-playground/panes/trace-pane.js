class TracePane {
    constructor(containerElement, onSelect = null) {
        this.container = containerElement;
        this.onSelect = onSelect;
        this.activeSeq = null;
        this.events = [];
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
        this.events.push(event);
        this.events.sort((a, b) => Number.parseInt(a.seq, 10) - Number.parseInt(b.seq, 10));
        this.render();
    }

    selectSeq(seq) {
        this.activeSeq = Number.parseInt(seq, 10);
        if (Number.isNaN(this.activeSeq)) {
            this.activeSeq = null;
        }
        this.render();
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
        headline.textContent = `${event.seq}. ${event.event_type}`;
        row.appendChild(headline);

        const detail = document.createElement("div");
        detail.className = "ap-row-meta";
        detail.textContent = `${event.stage} | ${event.status} | ${event.timestamp}`;
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
        if (this.events.length === 0) {
            const empty = document.createElement("div");
            empty.className = "ap-empty";
            empty.textContent = "No events recorded yet.";
            this.container.appendChild(empty);
            return;
        }

        for (const event of this.events) {
            this.container.appendChild(this._buildRow(event));
        }
    }
}

export { TracePane };
