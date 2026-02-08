import { diffObjects } from "../state-diff.js";

function summarizeDiff(diff) {
    const added = Object.keys(diff?.added || {}).length;
    const removed = Object.keys(diff?.removed || {}).length;
    const changed = Object.keys(diff?.changed || {}).length;
    return { added, removed, changed };
}

function createCard(title) {
    const card = document.createElement("section");
    card.className = "ap-card";
    const heading = document.createElement("div");
    heading.className = "ap-card-headline";
    heading.textContent = title;
    card.appendChild(heading);
    return card;
}

class StatePane {
    constructor(containerElement) {
        this.container = containerElement;
        this.run = null;
        this.snapshots = [];
        this.selectedEvent = null;
        this.render();
    }

    setRun(run) {
        this.run = run;
        this.render();
    }

    setSnapshots(snapshots) {
        this.snapshots = Array.isArray(snapshots) ? snapshots.slice() : [];
        this.render();
    }

    setSelectedEvent(event) {
        this.selectedEvent = event;
        this.render();
    }

    _snapshotForSeq(seq) {
        const target = Number.parseInt(seq, 10);
        if (Number.isNaN(target)) {
            return null;
        }
        const candidate = this.snapshots.find((snapshot) => Number.parseInt(snapshot.step_seq, 10) === target);
        return candidate || null;
    }

    _latestSnapshot() {
        if (!Array.isArray(this.snapshots) || this.snapshots.length === 0) {
            return null;
        }
        return this.snapshots[this.snapshots.length - 1];
    }

    _appendRunSummary() {
        const card = createCard("Run State");
        const values = document.createElement("div");
        values.className = "ap-kv-grid";
        const rows = {
            run_id: this.run.run_id || "-",
            mode: this.run.mode || "-",
            status: this.run.status || "-",
            created_at: this.run.created_at || "-",
            updated_at: this.run.updated_at || "-",
        };

        for (const [key, value] of Object.entries(rows)) {
            const keyNode = document.createElement("div");
            keyNode.className = "ap-kv-key";
            keyNode.textContent = key;
            values.appendChild(keyNode);

            const valueNode = document.createElement("div");
            valueNode.className = "ap-kv-value";
            valueNode.textContent = String(value);
            values.appendChild(valueNode);
        }
        card.appendChild(values);
        this.container.appendChild(card);
    }

    _appendSelectedSnapshot() {
        const selectedSeq = this.selectedEvent ? Number.parseInt(this.selectedEvent.seq, 10) : null;
        const currentSnapshot = selectedSeq ? this._snapshotForSeq(selectedSeq) : this._latestSnapshot();
        if (!currentSnapshot) {
            const empty = document.createElement("div");
            empty.className = "ap-empty";
            empty.textContent = "State snapshots will appear when stage events emit state metadata.";
            this.container.appendChild(empty);
            return;
        }

        const currentCard = createCard("Selected Snapshot");
        const meta = document.createElement("div");
        meta.className = "ap-row-meta";
        meta.textContent = `step ${currentSnapshot.step_seq} | ${currentSnapshot.stage || "-"} | ${currentSnapshot.timestamp || ""}`;
        currentCard.appendChild(meta);

        const state = (currentSnapshot && typeof currentSnapshot.state === "object") ? currentSnapshot.state : {};
        const detail = document.createElement("div");
        detail.className = "ap-row-summary";
        detail.textContent = String(state.detail || "No step detail provided.");
        currentCard.appendChild(detail);

        if (state.meta && typeof state.meta === "object" && Object.keys(state.meta).length > 0) {
            const metaDetails = document.createElement("details");
            metaDetails.className = "ap-details";
            const summary = document.createElement("summary");
            summary.textContent = "Step Metadata";
            metaDetails.appendChild(summary);
            const pre = document.createElement("pre");
            pre.className = "ap-json";
            pre.textContent = JSON.stringify(state.meta, null, 2);
            metaDetails.appendChild(pre);
            currentCard.appendChild(metaDetails);
        }

        const rawState = document.createElement("details");
        rawState.className = "ap-details";
        const rawSummary = document.createElement("summary");
        rawSummary.textContent = "Raw Snapshot JSON";
        rawState.appendChild(rawSummary);
        const rawBody = document.createElement("pre");
        rawBody.className = "ap-json";
        rawBody.textContent = JSON.stringify(currentSnapshot, null, 2);
        rawState.appendChild(rawBody);
        currentCard.appendChild(rawState);
        this.container.appendChild(currentCard);

        const previous = this._snapshotForSeq(Number.parseInt(currentSnapshot.step_seq, 10) - 1);
        if (!previous || !previous.state || !currentSnapshot.state) {
            return;
        }

        const diff = diffObjects(previous.state, currentSnapshot.state);
        const summary = summarizeDiff(diff);
        const diffCard = createCard("State Diff (Prev -> Current)");

        const summaryNode = document.createElement("div");
        summaryNode.className = "ap-row-summary";
        summaryNode.textContent = `added=${summary.added}, removed=${summary.removed}, changed=${summary.changed}`;
        diffCard.appendChild(summaryNode);

        const keyList = document.createElement("ul");
        keyList.className = "ap-list";
        const changedKeys = [
            ...Object.keys(diff.added || {}).map((key) => `added: ${key}`),
            ...Object.keys(diff.removed || {}).map((key) => `removed: ${key}`),
            ...Object.keys(diff.changed || {}).map((key) => `changed: ${key}`),
        ];
        for (const itemLabel of changedKeys.slice(0, 12)) {
            const item = document.createElement("li");
            item.textContent = itemLabel;
            keyList.appendChild(item);
        }
        if (changedKeys.length > 0) {
            diffCard.appendChild(keyList);
        }

        const diffDetails = document.createElement("details");
        diffDetails.className = "ap-details";
        const diffSummary = document.createElement("summary");
        diffSummary.textContent = "Raw Diff JSON";
        diffDetails.appendChild(diffSummary);
        const diffBody = document.createElement("pre");
        diffBody.className = "ap-json";
        diffBody.textContent = JSON.stringify({ diff }, null, 2);
        diffDetails.appendChild(diffBody);
        diffCard.appendChild(diffDetails);
        this.container.appendChild(diffCard);
    }

    render() {
        if (!this.container) {
            return;
        }

        this.container.innerHTML = "";
        if (!this.run) {
            const empty = document.createElement("div");
            empty.className = "ap-empty";
            empty.textContent = "State snapshots will appear during execution.";
            this.container.appendChild(empty);
            return;
        }

        this._appendRunSummary();
        this._appendSelectedSnapshot();
    }
}

export { StatePane };
