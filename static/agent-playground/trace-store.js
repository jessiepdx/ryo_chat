class TraceStore {
    constructor() {
        this.run = null;
        this.events = [];
        this.snapshots = [];
        this.artifacts = [];
        this.selectedSeq = null;
    }

    reset() {
        this.run = null;
        this.events = [];
        this.snapshots = [];
        this.artifacts = [];
        this.selectedSeq = null;
    }

    setRun(run) {
        this.run = run;
    }

    setEvents(events) {
        this.events = Array.isArray(events) ? events.slice() : [];
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
    }

    setSnapshots(snapshots) {
        this.snapshots = Array.isArray(snapshots) ? snapshots.slice() : [];
    }

    setArtifacts(artifacts) {
        this.artifacts = Array.isArray(artifacts) ? artifacts.slice() : [];
    }

    getEventBySeq(seq) {
        const normalized = Number.parseInt(seq, 10);
        return this.events.find((event) => Number.parseInt(event.seq, 10) === normalized) || null;
    }

    selectSeq(seq) {
        this.selectedSeq = Number.parseInt(seq, 10);
        if (Number.isNaN(this.selectedSeq)) {
            this.selectedSeq = null;
        }
    }

    getSelectedEvent() {
        if (this.selectedSeq === null) {
            return null;
        }
        return this.getEventBySeq(this.selectedSeq);
    }

    latestSeq() {
        if (!Array.isArray(this.events) || this.events.length === 0) {
            return 0;
        }
        return Number.parseInt(this.events[this.events.length - 1].seq, 10) || 0;
    }
}

export { TraceStore };
