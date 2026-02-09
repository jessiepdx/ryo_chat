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
    constructor(containerElement, handlers = {}) {
        this.container = containerElement;
        this.handlers = handlers && typeof handlers === "object" ? handlers : {};
        this.run = null;
        this.snapshots = [];
        this.selectedEvent = null;
        this.replayDraft = "";
        this.replayDraftSeq = null;
        this.replayDraftError = "";
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

    _notifyStatus(message) {
        const callback = this.handlers?.onStatus;
        if (typeof callback === "function") {
            callback(String(message || ""));
        }
    }

    _notifyReplayFromStep(stepSeq) {
        const callback = this.handlers?.onReplayFromStep;
        if (typeof callback !== "function") {
            return;
        }
        callback(Number.parseInt(stepSeq, 10));
    }

    _notifyReplayWithState(stepSeq, stateOverrides) {
        const callback = this.handlers?.onReplayWithState;
        if (typeof callback !== "function") {
            return;
        }
        callback(
            Number.parseInt(stepSeq, 10),
            (stateOverrides && typeof stateOverrides === "object") ? stateOverrides : {},
        );
    }

    _setReplayDraft(snapshot) {
        const seq = Number.parseInt(snapshot?.step_seq, 10);
        if (!Number.isFinite(seq)) {
            return;
        }
        if (this.replayDraftSeq === seq && this.replayDraft) {
            return;
        }
        const state = snapshot && typeof snapshot.state === "object" ? snapshot.state : {};
        this.replayDraft = JSON.stringify(state, null, 2);
        this.replayDraftSeq = seq;
        this.replayDraftError = "";
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

    _appendReplayControls(currentSnapshot) {
        const stepSeq = Number.parseInt(currentSnapshot?.step_seq, 10);
        if (!Number.isFinite(stepSeq)) {
            return;
        }
        this._setReplayDraft(currentSnapshot);

        const replayCard = createCard("Replay Controls");

        const helper = document.createElement("div");
        helper.className = "ap-row-meta";
        helper.textContent = `Replay source step: ${stepSeq}`;
        replayCard.appendChild(helper);

        const actionRow = document.createElement("div");
        actionRow.className = "ap-action-row";

        const replayStepButton = document.createElement("button");
        replayStepButton.type = "button";
        replayStepButton.textContent = "Replay From Selected Step";
        replayStepButton.addEventListener("click", () => {
            this._notifyStatus(`Replaying from step #${stepSeq}...`);
            this._notifyReplayFromStep(stepSeq);
        });
        actionRow.appendChild(replayStepButton);

        const replayEditedButton = document.createElement("button");
        replayEditedButton.type = "button";
        replayEditedButton.textContent = "Replay With Edited State";
        replayEditedButton.addEventListener("click", () => {
            let parsedState = {};
            try {
                parsedState = JSON.parse(this.replayDraft || "{}");
            } catch (_error) {
                this.replayDraftError = "Edited state must be valid JSON.";
                this.render();
                return;
            }
            if (!parsedState || typeof parsedState !== "object" || Array.isArray(parsedState)) {
                this.replayDraftError = "Edited state must be a JSON object.";
                this.render();
                return;
            }
            this.replayDraftError = "";
            this._notifyStatus(`Replaying from step #${stepSeq} with edited state...`);
            this._notifyReplayWithState(stepSeq, parsedState);
        });
        actionRow.appendChild(replayEditedButton);

        replayCard.appendChild(actionRow);

        const editor = document.createElement("textarea");
        editor.className = "ap-state-editor";
        editor.value = this.replayDraft || "{}";
        editor.addEventListener("input", () => {
            this.replayDraft = editor.value;
            this.replayDraftError = "";
        });
        replayCard.appendChild(editor);

        const editorMeta = document.createElement("div");
        editorMeta.className = `ap-row-meta ${this.replayDraftError ? "ap-error" : ""}`.trim();
        editorMeta.textContent = this.replayDraftError || "Edit JSON state overrides, then replay.";
        replayCard.appendChild(editorMeta);

        this.container.appendChild(replayCard);
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
        this._appendReplayControls(currentSnapshot);

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
