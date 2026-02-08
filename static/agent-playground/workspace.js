import { TraceStore } from "./trace-store.js";
import { SchemaFormRenderer } from "./schema-renderer/renderer.js";
import { cancelRun, replayRun, resumeRun } from "./replay-controls.js";
import { buildChatPayload } from "./modes/chat-mode.js";
import { buildWorkflowPayload } from "./modes/workflow-mode.js";
import { buildBatchPayload } from "./modes/batch-mode.js";
import { buildComparePayload } from "./modes/compare-mode.js";
import { buildReplayPayload } from "./modes/replay-mode.js";
import { ChatPane } from "./panes/chat-pane.js";
import { TracePane } from "./panes/trace-pane.js";
import { StatePane } from "./panes/state-pane.js";
import { ArtifactsPane } from "./panes/artifacts-pane.js";
import { InspectorPane } from "./panes/inspector-pane.js";

const modeBuilders = {
    chat: buildChatPayload,
    workflow: buildWorkflowPayload,
    batch: buildBatchPayload,
    compare: buildComparePayload,
    replay: buildReplayPayload,
};

class AgentPlaygroundApp {
    constructor(rootElement) {
        this.root = rootElement;
        this.apiBase = rootElement.dataset.apiBase || "/api/agent-playground";
        this.eventSource = null;
        this.store = new TraceStore();
        this.bootstrapData = null;

        this.chatPane = new ChatPane(document.querySelector("[data-pane='chat']"));
        this.tracePane = new TracePane(document.querySelector("[data-pane='trace']"), (event) => this.onTraceSelect(event));
        this.statePane = new StatePane(document.querySelector("[data-pane='state']"));
        this.artifactsPane = new ArtifactsPane(document.querySelector("[data-pane='artifacts']"));
        this.inspectorPane = new InspectorPane(document.querySelector("[data-pane='inspector']"));

        this.refs = {
            modeSelect: document.getElementById("ap-run-mode"),
            runStatus: document.getElementById("ap-run-status"),
            runID: document.getElementById("ap-run-id"),
            input: document.getElementById("ap-input"),
            runBtn: document.getElementById("ap-run-btn"),
            startBtn: document.getElementById("ap-start-btn"),
            refreshBtn: document.getElementById("ap-refresh-btn"),
            cancelBtn: document.getElementById("ap-cancel-btn"),
            replayBtn: document.getElementById("ap-replay-btn"),
            resumeBtn: document.getElementById("ap-resume-btn"),
            schemaForm: document.getElementById("ap-schema-form"),
            schemaCapability: document.getElementById("ap-schema-capability"),
        };

        this.optionsRenderer = null;
        this.schemaSources = new Map();
    }

    setStatus(text) {
        if (!this.refs.runStatus) {
            return;
        }
        this.refs.runStatus.textContent = String(text || "Idle");
    }

    setRunID(runID) {
        if (!this.refs.runID) {
            return;
        }
        this.refs.runID.textContent = runID || "-";
    }

    activeRun() {
        return this.store.run;
    }

    activeRunID() {
        return this.activeRun()?.run_id || null;
    }

    async loadBootstrap() {
        const response = await fetch(`${this.apiBase}/bootstrap`);
        const payload = await response.json();
        if (!response.ok || payload.status !== "ok") {
            throw new Error(payload.message || "Failed to load playground bootstrap.");
        }
        this.bootstrapData = payload;

        this.populateRunModes(payload.run_modes || []);
        this.configureSchemaForm(payload.manifest || {});

        const initialRuns = Array.isArray(payload.runs) ? payload.runs : [];
        if (initialRuns.length > 0) {
            const latest = initialRuns[0];
            this.store.setRun(latest);
            this.statePane.setRun(latest);
            this.setRunID(latest.run_id);
            this.setStatus(latest.status || "idle");
            await this.refreshRunDetail(latest.run_id);
            if (latest.status === "running") {
                this.connectStream(latest.run_id);
            }
        } else {
            this.setStatus("Idle");
        }
    }

    populateRunModes(runModes) {
        if (!this.refs.modeSelect) {
            return;
        }

        this.refs.modeSelect.innerHTML = "";
        const modes = Array.isArray(runModes) && runModes.length > 0
            ? runModes
            : [
                { id: "chat", label: "Chat" },
                { id: "workflow", label: "Workflow" },
                { id: "batch", label: "Batch" },
                { id: "compare", label: "Compare" },
                { id: "replay", label: "Replay" },
            ];

        for (const mode of modes) {
            const option = document.createElement("option");
            option.value = mode.id;
            option.textContent = mode.label || mode.id;
            this.refs.modeSelect.appendChild(option);
        }
    }

    configureSchemaForm(manifest) {
        if (!this.refs.schemaForm || !this.refs.schemaCapability) {
            return;
        }

        const capabilities = Array.isArray(manifest.capabilities) ? manifest.capabilities : [];
        this.schemaSources.clear();
        this.refs.schemaCapability.innerHTML = "";

        const addSource = (id, label, schema, seed = {}) => {
            if (!schema || typeof schema !== "object") {
                return;
            }
            this.schemaSources.set(id, { label, schema, seed });
            const option = document.createElement("option");
            option.value = id;
            option.textContent = label;
            this.refs.schemaCapability.appendChild(option);
        };

        const modelsCapability = capabilities.find((capability) => capability?.id === "models.list");
        addSource(
            "models.list",
            "Model Parameters",
            modelsCapability?.schema || {
                type: "object",
                properties: {
                    model_requested: { type: "string", description: "Preferred model for this run" },
                },
            },
            {},
        );

        const runCapability = capabilities.find((capability) => capability?.id === "runs.lifecycle");
        if (runCapability?.schema) {
            addSource("runs.lifecycle", "Run Lifecycle", runCapability.schema, {});
        }

        const toolsCapability = capabilities.find((capability) => capability?.id === "tools.list");
        const tools = Array.isArray(toolsCapability?.items) ? toolsCapability.items : [];
        for (const tool of tools) {
            if (!tool || !tool.name || !tool.input_schema) {
                continue;
            }
            addSource(`tool:${tool.name}`, `Tool Args: ${tool.name}`, tool.input_schema, {});
        }

        if (this.schemaSources.size === 0) {
            return;
        }

        const renderSelected = () => {
            const sourceID = this.refs.schemaCapability.value;
            const source = this.schemaSources.get(sourceID);
            if (!source) {
                return;
            }
            this.optionsRenderer = new SchemaFormRenderer(this.refs.schemaForm, source.schema, source.seed);
            this.optionsRenderer.render();
        };

        this.refs.schemaCapability.onchange = renderSelected;
        renderSelected();
    }

    bindEvents() {
        this.refs.runBtn?.addEventListener("click", () => this.startRun());
        this.refs.startBtn?.addEventListener("click", () => this.startRun());
        this.refs.refreshBtn?.addEventListener("click", () => {
            const runID = this.activeRunID();
            if (runID) {
                this.refreshRunDetail(runID);
            }
        });
        this.refs.cancelBtn?.addEventListener("click", () => this.cancelActiveRun());
        this.refs.replayBtn?.addEventListener("click", () => this.replayActiveRun());
        this.refs.resumeBtn?.addEventListener("click", () => this.resumeActiveRun());

        this.refs.input?.addEventListener("keydown", (event) => {
            if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                event.preventDefault();
                this.startRun();
            }
        });
    }

    getMode() {
        const mode = this.refs.modeSelect?.value || "chat";
        return modeBuilders[mode] ? mode : "chat";
    }

    optionValues() {
        if (!this.optionsRenderer) {
            return {};
        }
        const validation = this.optionsRenderer.validate();
        this.inspectorPane.setSchemaValidation(validation);
        if (!validation.valid) {
            return null;
        }
        const value = validation.value;
        if (this.refs.schemaCapability?.value) {
            value._schema_capability = this.refs.schemaCapability.value;
        }
        return value;
    }

    async startRun() {
        const mode = this.getMode();
        const inputText = String(this.refs.input?.value || "").trim();
        const options = this.optionValues();
        if (options === null) {
            this.setStatus("Invalid options");
            return;
        }

        const builder = modeBuilders[mode] || buildChatPayload;
        const payload = builder(inputText, options);

        if (["chat", "workflow", "compare"].includes(mode) && !payload.message) {
            this.setStatus("Message required");
            return;
        }

        if (mode === "batch" && (!Array.isArray(payload.batch_inputs) || payload.batch_inputs.length === 0)) {
            this.setStatus("Batch input required");
            return;
        }

        this.setStatus("Starting run...");

        const response = await fetch(`${this.apiBase}/runs`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });

        const runPayload = await response.json();
        if (!response.ok || runPayload.status !== "ok") {
            this.setStatus(runPayload.message || "Run start failed");
            this.chatPane.addSystemMessage(`Run failed: ${runPayload.message || response.statusText}`);
            return;
        }

        const run = runPayload.run;
        this.store.reset();
        this.store.setRun(run);
        this.statePane.setRun(run);
        this.setRunID(run.run_id);
        this.setStatus(run.status || "queued");

        if (payload.message) {
            this.chatPane.addUserMessage(payload.message);
        }

        await this.refreshRunDetail(run.run_id);
        this.connectStream(run.run_id);
    }

    onTraceSelect(event) {
        if (!event || typeof event !== "object") {
            return;
        }
        this.store.selectSeq(event.seq);
        this.tracePane.selectSeq(event.seq);
        this.statePane.setSelectedEvent(event);
        this.inspectorPane.setEvent(event);
    }

    handleIncomingEvent(event) {
        this.store.appendEvent(event);
        this.tracePane.appendEvent(event);

        const eventType = String(event.event_type || "");
        if (eventType === "run.token") {
            this.chatPane.appendAssistantChunk(event.payload?.chunk || "");
        }
        if (eventType === "run.completed") {
            const responseText =
                event.payload?.result?.response ||
                event.payload?.response_preview ||
                "Run completed.";
            this.chatPane.finalizeAssistantMessage(String(responseText || ""));
            this.setStatus("Completed");
        }
        if (eventType === "run.failed") {
            this.setStatus("Failed");
            this.chatPane.addSystemMessage(`Run failed: ${event.payload?.error || "unknown error"}`);
        }
        if (eventType === "run.cancelled") {
            this.setStatus("Cancelled");
            this.chatPane.addSystemMessage("Run cancelled.");
        }

        if (this.store.selectedSeq === null) {
            this.onTraceSelect(event);
        }

        const seq = Number.parseInt(event.seq, 10);
        if (!Number.isNaN(seq) && seq % 5 === 0) {
            this.refreshRunDetail(event.run_id);
        }
    }

    connectStream(runID) {
        if (!runID) {
            return;
        }

        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }

        const afterSeq = this.store.latestSeq();
        const streamURL = `${this.apiBase}/runs/${encodeURIComponent(runID)}/stream?after_seq=${afterSeq}`;
        const eventSource = new EventSource(streamURL);

        eventSource.onmessage = (messageEvent) => {
            try {
                const event = JSON.parse(messageEvent.data);
                this.handleIncomingEvent(event);
            } catch (_error) {
                return;
            }
        };

        eventSource.addEventListener("done", async (messageEvent) => {
            try {
                const payload = JSON.parse(messageEvent.data || "{}");
                if (payload.status) {
                    this.setStatus(payload.status);
                }
            } catch (_error) {
                return;
            } finally {
                eventSource.close();
                if (this.eventSource === eventSource) {
                    this.eventSource = null;
                }
                await this.refreshRunDetail(runID);
            }
        });

        eventSource.onerror = () => {
            this.setStatus("Stream disconnected");
        };

        this.eventSource = eventSource;
    }

    async refreshRunDetail(runID) {
        if (!runID) {
            return;
        }

        const response = await fetch(`${this.apiBase}/runs/${encodeURIComponent(runID)}`);
        const payload = await response.json();
        if (!response.ok || payload.status !== "ok") {
            return;
        }

        this.store.setRun(payload.run);
        this.store.setEvents(payload.events || []);
        this.store.setSnapshots(payload.snapshots || []);
        this.store.setArtifacts(payload.artifacts || []);

        this.tracePane.setEvents(this.store.events);
        this.statePane.setRun(payload.run);
        this.statePane.setSnapshots(this.store.snapshots);
        this.artifactsPane.setArtifacts(this.store.artifacts);
        this.inspectorPane.setMetrics({
            event_count: this.store.events.length,
            snapshot_count: this.store.snapshots.length,
            artifact_count: this.store.artifacts.length,
        });

        this.setRunID(payload.run.run_id);
        this.setStatus(payload.run.status || "idle");

        const selected = this.store.getSelectedEvent();
        if (selected) {
            this.tracePane.selectSeq(selected.seq);
            this.statePane.setSelectedEvent(selected);
            this.inspectorPane.setEvent(selected);
        }
    }

    async cancelActiveRun() {
        const runID = this.activeRunID();
        if (!runID) {
            return;
        }
        const payload = await cancelRun(this.apiBase, runID);
        if (payload?.run) {
            await this.refreshRunDetail(payload.run.run_id);
        }
    }

    async replayActiveRun() {
        const runID = this.activeRunID();
        if (!runID) {
            return;
        }
        const selected = this.store.getSelectedEvent();
        const stepSeq = selected?.seq ? Number.parseInt(selected.seq, 10) : undefined;
        const payload = await replayRun(this.apiBase, runID, {
            replay_from_seq: Number.isNaN(stepSeq) ? undefined : stepSeq,
        });
        if (payload?.run?.run_id) {
            this.store.reset();
            this.setRunID(payload.run.run_id);
            this.setStatus(payload.run.status || "queued");
            await this.refreshRunDetail(payload.run.run_id);
            this.connectStream(payload.run.run_id);
        }
    }

    async resumeActiveRun() {
        const runID = this.activeRunID();
        if (!runID) {
            return;
        }
        const payload = await resumeRun(this.apiBase, runID);
        if (payload?.run?.run_id) {
            this.store.reset();
            this.setRunID(payload.run.run_id);
            this.setStatus(payload.run.status || "queued");
            await this.refreshRunDetail(payload.run.run_id);
            this.connectStream(payload.run.run_id);
        }
    }

    async init() {
        this.bindEvents();
        await this.loadBootstrap();
    }
}

async function bootstrapPlayground() {
    const root = document.getElementById("ap-workspace");
    if (!root) {
        return;
    }

    const app = new AgentPlaygroundApp(root);
    try {
        await app.init();
    } catch (error) {
        const message = `Playground init failed: ${error}`;
        console.error(message);
        app.chatPane.addSystemMessage(message);
        app.setStatus("Init failed");
    }
}

document.addEventListener("DOMContentLoaded", bootstrapPlayground);
