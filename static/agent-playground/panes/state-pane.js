import { diffObjects } from "../state-diff.js";

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

        const runNode = document.createElement("pre");
        runNode.className = "ap-json";
        runNode.textContent = JSON.stringify(
            {
                run_id: this.run.run_id,
                mode: this.run.mode,
                status: this.run.status,
                updated_at: this.run.updated_at,
            },
            null,
            2,
        );
        this.container.appendChild(runNode);

        if (!this.selectedEvent) {
            return;
        }

        const currentSnapshot = this._snapshotForSeq(this.selectedEvent.seq);
        if (!currentSnapshot) {
            return;
        }

        const selectedNode = document.createElement("pre");
        selectedNode.className = "ap-json";
        selectedNode.textContent = JSON.stringify(currentSnapshot, null, 2);
        this.container.appendChild(selectedNode);

        const previous = this._snapshotForSeq(Number.parseInt(this.selectedEvent.seq, 10) - 1);
        if (previous && previous.state && currentSnapshot.state) {
            const diff = diffObjects(previous.state, currentSnapshot.state);
            const diffNode = document.createElement("pre");
            diffNode.className = "ap-json";
            diffNode.textContent = JSON.stringify({ diff }, null, 2);
            this.container.appendChild(diffNode);
        }
    }
}

export { StatePane };
