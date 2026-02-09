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
import { AgentBuilder } from "./agent-builder.js";
import { ToolRegistryView } from "./tools/tool-registry-view.js";
import { SandboxPolicyEditor } from "./tools/sandbox-policy-editor.js";
import { ApprovalQueueView } from "./tools/approval-queue.js";
import { ToolHarnessView } from "./tools/tool-harness.js";

const modeBuilders = {
    chat: buildChatPayload,
    workflow: buildWorkflowPayload,
    batch: buildBatchPayload,
    compare: buildComparePayload,
    replay: buildReplayPayload,
};

const fallbackRunModes = [
    { id: "chat", label: "Chat" },
    { id: "workflow", label: "Workflow" },
    { id: "batch", label: "Batch" },
    { id: "compare", label: "Compare" },
    { id: "replay", label: "Replay" },
];

const workspacePrefsStorageKey = "ryo.agent-playground.workspace-prefs.v1";
const runModeStorageKey = "ryo.agent-playground.last-mode.v1";
const modeGuidance = Object.freeze({
    chat: {
        helper: "Chat mode: Ask one question and run a single conversational pass.",
        placeholder: "Enter a user prompt.",
    },
    workflow: {
        helper: "Workflow mode: Use multiple lines to define sequential stage intents.",
        placeholder: "Enter one workflow step per line (line 1 = step_1, line 2 = step_2, etc).",
    },
    batch: {
        helper: "Batch mode: Submit one input per line and review row-by-row outcomes.",
        placeholder: "Enter one batch input per line.",
    },
    compare: {
        helper: "Compare mode: Run one prompt across multiple models and inspect output diffs.",
        placeholder: "Enter one prompt to run across compare models.",
    },
    replay: {
        helper: "Replay mode: Re-run a prior run or replay from a selected step/snapshot.",
        placeholder: "Optional prompt override for replay context.",
    },
});
const paneDefinitions = Object.freeze([
    { key: "schema", label: "Schema", elementID: "ap-schema-pane" },
    { key: "chat", label: "Chat", elementID: "ap-chat-pane" },
    { key: "trace", label: "Trace", elementID: "ap-trace-pane" },
    { key: "state", label: "State", elementID: "ap-state-pane" },
    { key: "artifacts", label: "Artifacts", elementID: "ap-artifacts-pane" },
    { key: "inspector", label: "Inspector", elementID: "ap-inspector-pane" },
]);
const layoutPresets = new Set(["balanced", "chat_focus", "analysis_focus"]);

function normalizeRunLabel(run) {
    if (!run || typeof run !== "object") {
        return "-";
    }
    const runID = String(run.run_id || "");
    const mode = String(run.mode || "unknown");
    const status = String(run.status || "unknown");
    const updated = String(run.updated_at || run.created_at || "").replace("T", " ").replace("Z", "");
    const shortID = runID ? runID.slice(0, 8) : "run";
    return `${shortID} | ${mode} | ${status} | ${updated || "-"}`;
}

function countStatus(metrics, statusKey) {
    const counts = (metrics && typeof metrics.status_counts === "object") ? metrics.status_counts : {};
    return Number.parseInt(counts?.[statusKey], 10) || 0;
}

function shortText(value, maxLength = 220) {
    const text = String(value || "").replace(/\s+/g, " ").trim();
    if (!text) {
        return "-";
    }
    if (text.length <= maxLength) {
        return text;
    }
    return `${text.slice(0, Math.max(0, maxLength - 3))}...`;
}

class AgentPlaygroundApp {
    constructor(rootElement) {
        this.root = rootElement;
        this.apiBase = rootElement.dataset.apiBase || "/api/agent-playground";
        this.eventSource = null;
        this.store = new TraceStore();
        this.bootstrapData = null;
        this.runHistory = [];
        this.panes = new Map(
            paneDefinitions.map((pane) => [
                pane.key,
                { ...pane, element: document.getElementById(pane.elementID) },
            ]),
        );

        this.chatPane = new ChatPane(document.querySelector("[data-pane='chat']"));
        this.tracePane = new TracePane(document.querySelector("[data-pane='trace']"), (event) => this.onTraceSelect(event));
        this.statePane = new StatePane(document.querySelector("[data-pane='state']"), {
            onStatus: (message) => this.setStatus(message),
            onReplayFromStep: (stepSeq) => this.replayFromStep(stepSeq),
            onReplayWithState: (stepSeq, stateOverrides) => this.replayWithState(stepSeq, stateOverrides),
        });
        this.artifactsPane = new ArtifactsPane(document.querySelector("[data-pane='artifacts']"));
        this.inspectorPane = new InspectorPane(document.querySelector("[data-pane='inspector']"));

        this.refs = {
            modeSelect: document.getElementById("ap-run-mode"),
            runHistory: document.getElementById("ap-run-history"),
            runStatus: document.getElementById("ap-run-status"),
            runID: document.getElementById("ap-run-id"),
            runModeActive: document.getElementById("ap-run-mode-active"),
            modeHelper: document.getElementById("ap-mode-helper"),
            input: document.getElementById("ap-input"),
            runBtn: document.getElementById("ap-run-btn"),
            startBtn: document.getElementById("ap-start-btn"),
            refreshBtn: document.getElementById("ap-refresh-btn"),
            cancelBtn: document.getElementById("ap-cancel-btn"),
            replayBtn: document.getElementById("ap-replay-btn"),
            resumeBtn: document.getElementById("ap-resume-btn"),
            schemaForm: document.getElementById("ap-schema-form"),
            schemaCapability: document.getElementById("ap-schema-capability"),
            paneControls: document.getElementById("ap-pane-controls"),
            layoutPreset: document.getElementById("ap-layout-preset"),
            layoutReset: document.getElementById("ap-layout-reset"),
            modelRequestedInput: document.getElementById("ap-model-requested-input"),
            modelRequestedSelect: document.getElementById("ap-model-requested-select"),
            modelSourceHint: document.getElementById("ap-model-source-hint"),
            sideRunID: document.getElementById("ap-side-run-id"),
            sideRunStatus: document.getElementById("ap-side-run-status"),
            sideRunMode: document.getElementById("ap-side-run-mode"),
            sideRunModel: document.getElementById("ap-side-run-model"),
            sideRunEvents: document.getElementById("ap-side-run-events"),
            sideEventSeq: document.getElementById("ap-side-event-seq"),
            sideEventType: document.getElementById("ap-side-event-type"),
            sideEventStage: document.getElementById("ap-side-event-stage"),
            sideEventStatus: document.getElementById("ap-side-event-status"),
            sideEventSummary: document.getElementById("ap-side-event-summary"),
            metricTotal: document.getElementById("ap-metric-total"),
            metricRunning: document.getElementById("ap-metric-running"),
            metricAvg: document.getElementById("ap-metric-avg"),
            metricCompleted: document.getElementById("ap-metric-completed"),
            metricFailed: document.getElementById("ap-metric-failed"),
            agentDefinitionSelect: document.getElementById("ap-agent-definition-select"),
            agentDefinitionVersion: document.getElementById("ap-agent-definition-version"),
            agentDefinitionName: document.getElementById("ap-agent-definition-name"),
            agentDefinitionSummary: document.getElementById("ap-agent-change-summary"),
            agentDefinitionJSON: document.getElementById("ap-agent-definition-json"),
            agentDefinitionStatus: document.getElementById("ap-agent-definition-status"),
            agentDefinitionHistory: document.getElementById("ap-agent-definition-history"),
            agentDefinitionRefresh: document.getElementById("ap-agent-definition-refresh"),
            agentDefinitionNew: document.getElementById("ap-agent-definition-new"),
            agentDefinitionSaveVersion: document.getElementById("ap-agent-definition-save-version"),
            agentDefinitionRollback: document.getElementById("ap-agent-definition-rollback"),
            agentDefinitionExport: document.getElementById("ap-agent-definition-export"),
            agentDefinitionImport: document.getElementById("ap-agent-definition-import"),
            toolRegistrySelect: document.getElementById("ap-tool-registry-select"),
            toolRegistryStatus: document.getElementById("ap-tool-registry-status"),
            toolRegistryWriteHint: document.getElementById("ap-tool-registry-write-hint"),
            toolRegistryRefresh: document.getElementById("ap-tool-registry-refresh"),
            toolRegistryNew: document.getElementById("ap-tool-registry-new"),
            toolRegistrySave: document.getElementById("ap-tool-registry-save"),
            toolRegistryDelete: document.getElementById("ap-tool-registry-delete"),
            toolRegistrySchema: document.getElementById("ap-tool-input-schema-json"),
            toolRegistryStaticResult: document.getElementById("ap-tool-static-result-json"),
            toolRegistryMetadataForm: document.getElementById("ap-tool-metadata-form"),
            toolRegistryArgPreviewForm: document.getElementById("ap-tool-arg-preview-form"),
            toolRegistrySelectedSummary: document.getElementById("ap-tool-selected-summary"),
            sandboxToolSelect: document.getElementById("ap-sandbox-tool-select"),
            sandboxToolName: document.getElementById("ap-sandbox-tool-name"),
            sandboxSourceLabel: document.getElementById("ap-sandbox-source-label"),
            sandboxSideEffect: document.getElementById("ap-sandbox-side-effect"),
            sandboxApprovalRequired: document.getElementById("ap-sandbox-approval-required"),
            sandboxDryRun: document.getElementById("ap-sandbox-dry-run"),
            sandboxApprovalTimeout: document.getElementById("ap-sandbox-approval-timeout"),
            sandboxTimeoutCeiling: document.getElementById("ap-sandbox-timeout-ceiling"),
            sandboxMaxMemory: document.getElementById("ap-sandbox-max-memory"),
            sandboxNetworkEnabled: document.getElementById("ap-sandbox-network-enabled"),
            sandboxNetworkAllowlist: document.getElementById("ap-sandbox-network-allowlist"),
            sandboxFilesystemMode: document.getElementById("ap-sandbox-filesystem-mode"),
            sandboxFilesystemAllowedPaths: document.getElementById("ap-sandbox-filesystem-allowed-paths"),
            sandboxWriteHint: document.getElementById("ap-sandbox-write-hint"),
            sandboxStatus: document.getElementById("ap-sandbox-status"),
            sandboxRefresh: document.getElementById("ap-sandbox-refresh"),
            sandboxNew: document.getElementById("ap-sandbox-new"),
            sandboxSave: document.getElementById("ap-sandbox-save"),
            sandboxDelete: document.getElementById("ap-sandbox-delete"),
            approvalQueueList: document.getElementById("ap-approval-queue-list"),
            approvalWriteHint: document.getElementById("ap-approval-write-hint"),
            approvalStatus: document.getElementById("ap-approval-status"),
            approvalRefresh: document.getElementById("ap-approval-refresh"),
            toolHarnessCaseSelect: document.getElementById("ap-tool-harness-case-select"),
            toolHarnessToolSelect: document.getElementById("ap-tool-harness-tool-select"),
            toolHarnessFixtureName: document.getElementById("ap-tool-harness-fixture-name"),
            toolHarnessDescription: document.getElementById("ap-tool-harness-description"),
            toolHarnessMode: document.getElementById("ap-tool-harness-mode"),
            toolHarnessInputJSON: document.getElementById("ap-tool-harness-input-json"),
            toolHarnessWriteHint: document.getElementById("ap-tool-harness-write-hint"),
            toolHarnessStatus: document.getElementById("ap-tool-harness-status"),
            toolHarnessReport: document.getElementById("ap-tool-harness-report"),
            toolHarnessRefresh: document.getElementById("ap-tool-harness-refresh"),
            toolHarnessNew: document.getElementById("ap-tool-harness-new"),
            toolHarnessSave: document.getElementById("ap-tool-harness-save"),
            toolHarnessRun: document.getElementById("ap-tool-harness-run"),
            toolHarnessSaveGolden: document.getElementById("ap-tool-harness-save-golden"),
            toolHarnessSaveContract: document.getElementById("ap-tool-harness-save-contract"),
            toolHarnessDelete: document.getElementById("ap-tool-harness-delete"),
        };

        this.optionsRenderer = null;
        this.schemaSources = new Map();
        this.modelInventory = [];
        this.agentBuilder = new AgentBuilder(
            this.apiBase,
            {
                definitionSelect: this.refs.agentDefinitionSelect,
                versionSelect: this.refs.agentDefinitionVersion,
                nameInput: this.refs.agentDefinitionName,
                summaryInput: this.refs.agentDefinitionSummary,
                definitionJSON: this.refs.agentDefinitionJSON,
                status: this.refs.agentDefinitionStatus,
                history: this.refs.agentDefinitionHistory,
                refreshBtn: this.refs.agentDefinitionRefresh,
                newBtn: this.refs.agentDefinitionNew,
                saveVersionBtn: this.refs.agentDefinitionSaveVersion,
                rollbackBtn: this.refs.agentDefinitionRollback,
                exportBtn: this.refs.agentDefinitionExport,
                importBtn: this.refs.agentDefinitionImport,
            },
            {
                onStatus: (message) => this.setStatus(message),
            },
        );
        this.toolRegistryView = new ToolRegistryView(
            this.apiBase,
            {
                toolSelect: this.refs.toolRegistrySelect,
                status: this.refs.toolRegistryStatus,
                writeHint: this.refs.toolRegistryWriteHint,
                refreshBtn: this.refs.toolRegistryRefresh,
                newBtn: this.refs.toolRegistryNew,
                saveBtn: this.refs.toolRegistrySave,
                deleteBtn: this.refs.toolRegistryDelete,
                schemaJSON: this.refs.toolRegistrySchema,
                staticResultJSON: this.refs.toolRegistryStaticResult,
                metadataForm: this.refs.toolRegistryMetadataForm,
                argPreviewForm: this.refs.toolRegistryArgPreviewForm,
                selectedSummary: this.refs.toolRegistrySelectedSummary,
            },
            {
                onStatus: (message) => this.setStatus(message),
            },
        );
        this.sandboxPolicyEditor = new SandboxPolicyEditor(
            this.apiBase,
            {
                toolSelect: this.refs.sandboxToolSelect,
                toolNameInput: this.refs.sandboxToolName,
                sourceLabel: this.refs.sandboxSourceLabel,
                sideEffectClass: this.refs.sandboxSideEffect,
                approvalRequired: this.refs.sandboxApprovalRequired,
                dryRunEnabled: this.refs.sandboxDryRun,
                approvalTimeoutSeconds: this.refs.sandboxApprovalTimeout,
                timeoutCeilingSeconds: this.refs.sandboxTimeoutCeiling,
                maxMemoryMB: this.refs.sandboxMaxMemory,
                networkEnabled: this.refs.sandboxNetworkEnabled,
                networkAllowlistDomains: this.refs.sandboxNetworkAllowlist,
                filesystemMode: this.refs.sandboxFilesystemMode,
                filesystemAllowedPaths: this.refs.sandboxFilesystemAllowedPaths,
                writeHint: this.refs.sandboxWriteHint,
                status: this.refs.sandboxStatus,
                refreshBtn: this.refs.sandboxRefresh,
                newBtn: this.refs.sandboxNew,
                saveBtn: this.refs.sandboxSave,
                deleteBtn: this.refs.sandboxDelete,
            },
            {
                onStatus: (message) => this.setStatus(message),
            },
        );
        this.approvalQueueView = new ApprovalQueueView(
            this.apiBase,
            {
                list: this.refs.approvalQueueList,
                writeHint: this.refs.approvalWriteHint,
                status: this.refs.approvalStatus,
                refreshBtn: this.refs.approvalRefresh,
            },
            {
                onStatus: (message) => this.setStatus(message),
            },
        );
        this.toolHarnessView = new ToolHarnessView(
            this.apiBase,
            {
                caseSelect: this.refs.toolHarnessCaseSelect,
                toolSelect: this.refs.toolHarnessToolSelect,
                fixtureName: this.refs.toolHarnessFixtureName,
                description: this.refs.toolHarnessDescription,
                mode: this.refs.toolHarnessMode,
                inputJSON: this.refs.toolHarnessInputJSON,
                writeHint: this.refs.toolHarnessWriteHint,
                status: this.refs.toolHarnessStatus,
                report: this.refs.toolHarnessReport,
                refreshBtn: this.refs.toolHarnessRefresh,
                newBtn: this.refs.toolHarnessNew,
                saveBtn: this.refs.toolHarnessSave,
                runBtn: this.refs.toolHarnessRun,
                saveGoldenBtn: this.refs.toolHarnessSaveGolden,
                saveContractBtn: this.refs.toolHarnessSaveContract,
                deleteBtn: this.refs.toolHarnessDelete,
            },
            {
                onStatus: (message) => this.setStatus(message),
            },
        );
        this.preferredRunMode = this._loadPreferredRunMode();
        this.workspacePrefs = this._loadWorkspacePrefs();
        this._bindPaneCollapseButtons();
        this._applyWorkspacePrefs();
    }

    _defaultWorkspacePrefs() {
        const visible = {};
        const collapsed = {};
        for (const definition of paneDefinitions) {
            visible[definition.key] = true;
            collapsed[definition.key] = false;
        }
        return {
            layout: "balanced",
            visible,
            collapsed,
        };
    }

    _normalizeWorkspacePrefs(rawPrefs) {
        const defaults = this._defaultWorkspacePrefs();
        const normalized = {
            layout: defaults.layout,
            visible: { ...defaults.visible },
            collapsed: { ...defaults.collapsed },
        };
        if (!rawPrefs || typeof rawPrefs !== "object") {
            return normalized;
        }

        const layout = String(rawPrefs.layout || "").trim();
        if (layoutPresets.has(layout)) {
            normalized.layout = layout;
        }

        for (const definition of paneDefinitions) {
            const key = definition.key;
            if (rawPrefs.visible && typeof rawPrefs.visible === "object" && key in rawPrefs.visible) {
                normalized.visible[key] = Boolean(rawPrefs.visible[key]);
            }
            if (rawPrefs.collapsed && typeof rawPrefs.collapsed === "object" && key in rawPrefs.collapsed) {
                normalized.collapsed[key] = Boolean(rawPrefs.collapsed[key]);
            }
            // Collapse now means hidden from layout. Keep legacy prefs compatible.
            if (normalized.collapsed[key]) {
                normalized.visible[key] = false;
            }
        }

        if (Object.values(normalized.visible).every((value) => !value)) {
            normalized.visible.chat = true;
        }

        return normalized;
    }

    _loadWorkspacePrefs() {
        try {
            const raw = window.localStorage.getItem(workspacePrefsStorageKey);
            if (!raw) {
                return this._defaultWorkspacePrefs();
            }
            const parsed = JSON.parse(raw);
            return this._normalizeWorkspacePrefs(parsed);
        } catch (_error) {
            return this._defaultWorkspacePrefs();
        }
    }

    _saveWorkspacePrefs() {
        try {
            window.localStorage.setItem(
                workspacePrefsStorageKey,
                JSON.stringify(this.workspacePrefs),
            );
        } catch (_error) {
            return;
        }
    }

    _loadPreferredRunMode() {
        try {
            const stored = String(window.localStorage.getItem(runModeStorageKey) || "").trim().toLowerCase();
            if (stored && modeBuilders[stored]) {
                return stored;
            }
        } catch (_error) {
            return "chat";
        }
        return "chat";
    }

    _savePreferredRunMode(mode) {
        const normalized = String(mode || "").trim().toLowerCase();
        if (!modeBuilders[normalized]) {
            return;
        }
        this.preferredRunMode = normalized;
        try {
            window.localStorage.setItem(runModeStorageKey, normalized);
        } catch (_error) {
            return;
        }
    }

    _paneVisibleCount() {
        return Object.values(this.workspacePrefs.visible).filter(Boolean).length;
    }

    _setLayoutPreset(layoutName) {
        const normalized = String(layoutName || "").trim();
        this.workspacePrefs.layout = layoutPresets.has(normalized) ? normalized : "balanced";
        this._applyWorkspacePrefs();
        this._saveWorkspacePrefs();
    }

    _togglePaneVisible(paneKey) {
        const key = String(paneKey || "").trim();
        if (!this.panes.has(key)) {
            return;
        }
        const currentlyVisible = Boolean(this.workspacePrefs.visible[key]);
        if (currentlyVisible && this._paneVisibleCount() <= 1) {
            this.setStatus("At least one pane must remain visible.");
            return;
        }
        this.workspacePrefs.visible[key] = !currentlyVisible;
        this.workspacePrefs.collapsed[key] = false;
        this._applyWorkspacePrefs();
        this._saveWorkspacePrefs();
    }

    _togglePaneCollapsed(paneKey) {
        const key = String(paneKey || "").trim();
        if (!this.panes.has(key)) {
            return;
        }
        const currentlyVisible = Boolean(this.workspacePrefs.visible[key]);
        if (!currentlyVisible) {
            return;
        }
        // "Collapse" should remove the pane from layout so other panes can expand.
        this.workspacePrefs.visible[key] = false;
        this.workspacePrefs.collapsed[key] = true;
        this._applyWorkspacePrefs();
        this._saveWorkspacePrefs();
    }

    _resetPaneLayout() {
        this.workspacePrefs = this._defaultWorkspacePrefs();
        this._applyWorkspacePrefs();
        this._saveWorkspacePrefs();
    }

    _bindPaneCollapseButtons() {
        const controls = document.querySelectorAll("[data-pane-collapse]");
        for (const control of controls) {
            control.addEventListener("click", () => {
                const paneKey = String(control.dataset.paneCollapse || "");
                this._togglePaneCollapsed(paneKey);
            });
        }
    }

    _renderPaneToggles() {
        const container = this.refs.paneControls;
        if (!container) {
            return;
        }

        container.innerHTML = "";
        const label = document.createElement("span");
        label.className = "ap-toolbar-label";
        label.textContent = "Panes";
        container.appendChild(label);

        for (const definition of paneDefinitions) {
            const key = definition.key;
            const visible = Boolean(this.workspacePrefs.visible[key]);
            const toggle = document.createElement("button");
            toggle.type = "button";
            toggle.className = `ap-pane-toggle ${visible ? "is-active" : ""}`.trim();
            toggle.dataset.paneToggle = key;
            toggle.textContent = definition.label;
            toggle.title = visible ? "Hide pane" : "Show pane";
            toggle.addEventListener("click", () => this._togglePaneVisible(key));
            container.appendChild(toggle);
        }
    }

    _applyWorkspacePrefs() {
        if (this.root) {
            this.root.dataset.layout = String(this.workspacePrefs.layout || "balanced");
        }

        if (this.refs.layoutPreset) {
            this.refs.layoutPreset.value = String(this.workspacePrefs.layout || "balanced");
        }

        for (const [key, pane] of this.panes.entries()) {
            if (!pane.element) {
                continue;
            }
            const visible = Boolean(this.workspacePrefs.visible[key]);
            const collapsed = Boolean(this.workspacePrefs.collapsed[key]);
            pane.element.classList.toggle("ap-pane-hidden", !visible);
            pane.element.classList.toggle("ap-pane-collapsed", visible && collapsed);

            const collapseButton = pane.element.querySelector("[data-pane-collapse]");
            if (collapseButton) {
                collapseButton.textContent = "Collapse";
                collapseButton.setAttribute("aria-expanded", visible ? "true" : "false");
                collapseButton.disabled = !visible;
            }
        }

        this._renderPaneToggles();
    }

    setStatus(text) {
        if (!this.refs.runStatus) {
            return;
        }
        this.refs.runStatus.textContent = String(text || "Idle");
    }

    setRunMeta(run) {
        if (this.refs.runID) {
            this.refs.runID.textContent = String(run?.run_id || "-");
        }
        if (this.refs.runModeActive) {
            this.refs.runModeActive.textContent = String(run?.mode || "-");
        }
        this.renderSideRunSummary(run || null);
    }

    activeRun() {
        return this.store.run;
    }

    activeRunID() {
        return this.activeRun()?.run_id || null;
    }

    setMetrics(metrics) {
        const total = Number.parseInt(metrics?.total_runs, 10) || 0;
        const running = Array.isArray(metrics?.active_runs) ? metrics.active_runs.length : countStatus(metrics, "running");
        const avgSeconds = Number(metrics?.average_run_seconds || 0);
        const completed = countStatus(metrics, "completed");
        const failed = countStatus(metrics, "failed");

        if (this.refs.metricTotal) {
            this.refs.metricTotal.textContent = String(total);
        }
        if (this.refs.metricRunning) {
            this.refs.metricRunning.textContent = String(running);
        }
        if (this.refs.metricAvg) {
            this.refs.metricAvg.textContent = String(Number.isFinite(avgSeconds) ? avgSeconds.toFixed(2) : "0.00");
        }
        if (this.refs.metricCompleted) {
            this.refs.metricCompleted.textContent = String(completed);
        }
        if (this.refs.metricFailed) {
            this.refs.metricFailed.textContent = String(failed);
        }
    }

    populateRunHistory(runs, activeRunID = null) {
        this.runHistory = Array.isArray(runs) ? runs.slice() : [];
        if (!this.refs.runHistory) {
            return;
        }

        this.refs.runHistory.innerHTML = "";
        if (this.runHistory.length === 0) {
            const option = document.createElement("option");
            option.value = "";
            option.textContent = "No runs yet";
            this.refs.runHistory.appendChild(option);
            this.refs.runHistory.disabled = true;
            return;
        }

        this.refs.runHistory.disabled = false;
        for (const run of this.runHistory) {
            const option = document.createElement("option");
            option.value = String(run.run_id || "");
            option.textContent = normalizeRunLabel(run);
            this.refs.runHistory.appendChild(option);
        }

        const preferred = String(activeRunID || this.activeRunID() || this.runHistory[0].run_id || "");
        if (preferred) {
            this.refs.runHistory.value = preferred;
        }
    }

    upsertRunHistory(run) {
        if (!run || typeof run !== "object" || !run.run_id) {
            return;
        }
        const runID = String(run.run_id);
        const nextRuns = this.runHistory.slice();
        const index = nextRuns.findIndex((item) => String(item.run_id) === runID);
        if (index >= 0) {
            nextRuns[index] = run;
        } else {
            nextRuns.unshift(run);
        }
        nextRuns.sort((left, right) => String(right.updated_at || "").localeCompare(String(left.updated_at || "")));
        this.populateRunHistory(nextRuns, runID);
    }

    async refreshRunsList(activeRunID = null) {
        const response = await fetch(`${this.apiBase}/runs?limit=40`);
        const payload = await response.json();
        if (!response.ok || payload.status !== "ok") {
            return;
        }
        this.populateRunHistory(payload.runs || [], activeRunID || this.activeRunID());
    }

    async refreshMetrics() {
        const response = await fetch(`${this.apiBase}/metrics`);
        const payload = await response.json();
        if (!response.ok || payload.status !== "ok") {
            return;
        }
        this.setMetrics(payload.metrics || {});
    }

    panelMetrics() {
        const run = this.store.run || {};
        return {
            event_count: this.store.events.length,
            snapshot_count: this.store.snapshots.length,
            artifact_count: this.store.artifacts.length,
            run_status: run.status || "idle",
            run_mode: run.mode || "-",
            run_updated_at: run.updated_at || run.created_at || null,
        };
    }

    requestedModelValue() {
        return String(this.refs.modelRequestedInput?.value || "").trim();
    }

    setRequestedModelValue(value) {
        const requested = String(value || "").trim();
        if (this.refs.modelRequestedInput) {
            this.refs.modelRequestedInput.value = requested;
        }
        this.syncModelSelectToInput();
    }

    syncModelSelectToInput() {
        if (!this.refs.modelRequestedSelect) {
            return;
        }
        const requested = this.requestedModelValue();
        if (!requested) {
            this.refs.modelRequestedSelect.value = "";
            return;
        }
        const exact = this.modelInventory.find((modelName) => modelName === requested);
        this.refs.modelRequestedSelect.value = exact ? exact : "__custom__";
    }

    populateModelSelect(models, defaultModel = "") {
        if (!this.refs.modelRequestedSelect) {
            return;
        }

        this.modelInventory = Array.isArray(models)
            ? [...new Set(models.map((item) => String(item || "").trim()).filter(Boolean))]
            : [];

        const select = this.refs.modelRequestedSelect;
        select.innerHTML = "";

        const defaultsOption = document.createElement("option");
        defaultsOption.value = "";
        defaultsOption.textContent = "Use Router Default";
        select.appendChild(defaultsOption);

        const customOption = document.createElement("option");
        customOption.value = "__custom__";
        customOption.textContent = "Custom (from text input)";
        select.appendChild(customOption);

        for (const modelName of this.modelInventory) {
            const option = document.createElement("option");
            option.value = modelName;
            option.textContent = modelName;
            select.appendChild(option);
        }

        if (!this.requestedModelValue() && defaultModel) {
            this.setRequestedModelValue(defaultModel);
        }
        this.syncModelSelectToInput();
    }

    bindModelRequestedControls() {
        this.refs.modelRequestedSelect?.addEventListener("change", () => {
            const selected = String(this.refs.modelRequestedSelect?.value || "");
            if (selected && selected !== "__custom__") {
                this.setRequestedModelValue(selected);
            }
            if (selected === "") {
                this.setRequestedModelValue("");
            }
        });

        this.refs.modelRequestedInput?.addEventListener("input", () => {
            this.syncModelSelectToInput();
        });
    }

    async loadModelInventory() {
        try {
            const response = await fetch(`${this.apiBase}/models`);
            const payload = await response.json();
            if (!response.ok || payload.status !== "ok") {
                throw new Error(payload.message || "Model API failed.");
            }

            const configured = payload.configured_models && typeof payload.configured_models === "object"
                ? Object.values(payload.configured_models)
                : [];
            const available = Array.isArray(payload.available_models) ? payload.available_models : [];
            const combined = [...new Set([...available, ...configured].map((item) => String(item || "").trim()).filter(Boolean))];
            const defaultRequested = String(payload.default_model_requested || configured[0] || "").trim();
            this.populateModelSelect(combined, defaultRequested);

            if (this.refs.modelSourceHint) {
                const host = String(payload.ollama_host || "-");
                const count = Number.parseInt(payload.available_count, 10) || combined.length;
                if (payload.error) {
                    this.refs.modelSourceHint.textContent = `Model probe warning (${host}): ${payload.error}`;
                } else {
                    this.refs.modelSourceHint.textContent = `Loaded ${count} model(s) from ${host}.`;
                }
            }
        } catch (error) {
            this.populateModelSelect([], this.requestedModelValue());
            if (this.refs.modelSourceHint) {
                this.refs.modelSourceHint.textContent = `Model inventory unavailable: ${error}`;
            }
        }
    }

    renderSideRunSummary(run) {
        if (this.refs.sideRunID) {
            this.refs.sideRunID.textContent = String(run?.run_id || "-");
        }
        if (this.refs.sideRunStatus) {
            this.refs.sideRunStatus.textContent = String(run?.status || "idle");
        }
        if (this.refs.sideRunMode) {
            this.refs.sideRunMode.textContent = String(run?.mode || "-");
        }
        if (this.refs.sideRunEvents) {
            this.refs.sideRunEvents.textContent = String(this.store.events.length || 0);
        }

        const runRequest = run && typeof run.request === "object" ? run.request : {};
        const runOptions = runRequest && typeof runRequest.options === "object" ? runRequest.options : {};
        const requestedModel = String(runOptions.model_requested || this.requestedModelValue() || "").trim();
        if (this.refs.sideRunModel) {
            this.refs.sideRunModel.textContent = requestedModel || "router default";
        }
    }

    renderSideEventSummary(event) {
        if (this.refs.sideEventSeq) {
            this.refs.sideEventSeq.textContent = String(event?.seq || "-");
        }
        if (this.refs.sideEventType) {
            this.refs.sideEventType.textContent = String(event?.event_type || "-");
        }
        if (this.refs.sideEventStage) {
            this.refs.sideEventStage.textContent = String(event?.stage || "-");
        }
        if (this.refs.sideEventStatus) {
            this.refs.sideEventStatus.textContent = String(event?.status || "-");
        }

        if (!this.refs.sideEventSummary) {
            return;
        }
        if (!event || typeof event !== "object") {
            this.refs.sideEventSummary.textContent = "Select a trace row to inspect payload details.";
            return;
        }

        const payload = event.payload && typeof event.payload === "object" ? event.payload : {};
        const keys = Object.keys(payload);
        if (keys.length === 0) {
            this.refs.sideEventSummary.textContent = "No payload details for this step.";
            return;
        }

        const headlineKey = ["detail", "error", "reason", "response_preview", "chunk"].find((key) => key in payload);
        if (headlineKey) {
            this.refs.sideEventSummary.textContent = shortText(payload[headlineKey], 260);
            return;
        }
        this.refs.sideEventSummary.textContent = shortText(JSON.stringify(payload), 260);
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
        this.setMetrics(payload.metrics || {});

        const initialRuns = Array.isArray(payload.runs) ? payload.runs : [];
        this.populateRunHistory(initialRuns);
        if (initialRuns.length > 0) {
            const latest = initialRuns[0];
            this.store.setRun(latest);
            this.statePane.setRun(latest);
            this.setRunMeta(latest);
            this.setStatus(latest.status || "idle");
            await this.refreshRunDetail(latest.run_id);
            if (latest.status === "running") {
                this.connectStream(latest.run_id);
            }
        } else {
            this.setStatus("Idle");
            this.setRunMeta(null);
            this.renderSideEventSummary(null);
        }
    }

    populateRunModes(runModes) {
        if (!this.refs.modeSelect) {
            return;
        }

        this.refs.modeSelect.innerHTML = "";
        const modes = Array.isArray(runModes) && runModes.length > 0 ? runModes : fallbackRunModes;

        for (const mode of modes) {
            const option = document.createElement("option");
            option.value = mode.id;
            option.textContent = mode.label || mode.id;
            this.refs.modeSelect.appendChild(option);
        }

        const fallbackMode = String(modes[0]?.id || "chat");
        const preferred = this.preferredRunMode && modeBuilders[this.preferredRunMode]
            ? this.preferredRunMode
            : fallbackMode;
        const hasPreferred = modes.some((mode) => String(mode.id) === preferred);
        this.refs.modeSelect.value = hasPreferred ? preferred : fallbackMode;
        this.applyModeGuidance(this.refs.modeSelect.value);
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

        const sandboxCapability = capabilities.find((capability) => capability?.id === "tools.sandbox");
        if (sandboxCapability?.schema) {
            addSource("tools.sandbox", "Tool Sandbox Policy", sandboxCapability.schema, {});
        }

        const approvalsCapability = capabilities.find((capability) => capability?.id === "tools.approvals");
        if (approvalsCapability?.schema) {
            addSource("tools.approvals", "Tool Approval Queue", approvalsCapability.schema, {});
        }

        const harnessCapability = capabilities.find((capability) => capability?.id === "tools.harness");
        if (harnessCapability?.schema) {
            addSource("tools.harness", "Tool Harness Fixture", harnessCapability.schema, {});
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
        this.refs.refreshBtn?.addEventListener("click", async () => {
            const runID = this.activeRunID();
            if (runID) {
                await this.refreshRunDetail(runID);
            }
            await this.refreshRunsList(runID);
            await this.refreshMetrics();
        });
        this.refs.cancelBtn?.addEventListener("click", () => this.cancelActiveRun());
        this.refs.replayBtn?.addEventListener("click", () => this.replayActiveRun());
        this.refs.resumeBtn?.addEventListener("click", () => this.resumeActiveRun());
        this.refs.layoutPreset?.addEventListener("change", () => {
            this._setLayoutPreset(this.refs.layoutPreset?.value || "balanced");
        });
        this.refs.layoutReset?.addEventListener("click", () => {
            this._resetPaneLayout();
        });
        this.refs.modeSelect?.addEventListener("change", () => {
            this._savePreferredRunMode(this.getMode());
            this.applyModeGuidance(this.getMode());
            if (!this.activeRunID() && this.refs.runModeActive) {
                this.refs.runModeActive.textContent = this.getMode();
            }
        });
        this.refs.runHistory?.addEventListener("change", () => {
            const runID = String(this.refs.runHistory.value || "").trim();
            if (!runID) {
                return;
            }
            this.refreshRunDetail(runID);
        });

        this.refs.input?.addEventListener("keydown", (event) => {
            if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                event.preventDefault();
                this.startRun();
            }
        });

        this.bindModelRequestedControls();
    }

    getMode() {
        const mode = this.refs.modeSelect?.value || "chat";
        return modeBuilders[mode] ? mode : "chat";
    }

    applyModeGuidance(mode = this.getMode()) {
        const normalized = modeBuilders[mode] ? mode : "chat";
        const guidance = modeGuidance[normalized] || modeGuidance.chat;

        if (this.refs.modeHelper) {
            this.refs.modeHelper.textContent = String(guidance.helper || modeGuidance.chat.helper);
        }
        if (this.refs.input) {
            this.refs.input.placeholder = String(guidance.placeholder || modeGuidance.chat.placeholder);
        }
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
        const requestedModel = this.requestedModelValue();
        if (requestedModel) {
            value.model_requested = requestedModel;
        }
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
        const agentBinding = this.agentBuilder?.getRunBinding ? this.agentBuilder.getRunBinding() : null;
        if (agentBinding && typeof agentBinding === "object") {
            if (agentBinding.definition_id) {
                payload.agent_definition_id = String(agentBinding.definition_id);
            }
            if (Number.isInteger(agentBinding.definition_version)) {
                payload.agent_definition_version = Number(agentBinding.definition_version);
            }
            if (agentBinding.definition && typeof agentBinding.definition === "object") {
                payload.agent_definition = agentBinding.definition;
            }
        }
        payload.options = payload.options && typeof payload.options === "object" ? payload.options : {};
        const requestedModel = this.requestedModelValue();
        if (requestedModel) {
            payload.options.model_requested = requestedModel;
        }

        if (["chat", "workflow", "compare"].includes(mode) && !payload.message) {
            this.setStatus("Message required");
            return;
        }

        if (mode === "batch" && (!Array.isArray(payload.batch_inputs) || payload.batch_inputs.length === 0)) {
            this.setStatus("Batch input required");
            return;
        }

        this.setStatus("Starting run...");
        this.chatPane.reset();

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
        this.setRunMeta(run);
        this.setStatus(run.status || "queued");
        this.upsertRunHistory(run);

        if (payload.message) {
            this.chatPane.addUserMessage(payload.message);
        }

        await this.refreshRunDetail(run.run_id);
        this.connectStream(run.run_id);
        this.refreshMetrics();
    }

    onTraceSelect(event) {
        if (!event || typeof event !== "object") {
            return;
        }
        this.store.selectSeq(event.seq);
        this.tracePane.selectSeq(event.seq);
        this.statePane.setSelectedEvent(event);
        this.inspectorPane.setEvent(event);
        this.renderSideEventSummary(event);
    }

    handleIncomingEvent(event) {
        this.store.appendEvent(event);
        this.tracePane.appendEvent(event);
        this.renderSideRunSummary(this.store.run);

        const eventType = String(event.event_type || "");
        if (eventType === "run.token") {
            this.chatPane.appendAssistantChunk(event.payload?.chunk || "");
        }
        if (eventType === "run.completed") {
            if (this.store.run) {
                this.store.run.status = "completed";
            }
            const responseText =
                event.payload?.result?.response ||
                event.payload?.response_preview ||
                "Run completed.";
            this.chatPane.finalizeAssistantMessage(String(responseText || ""));
            this.setStatus("Completed");
            this.refreshMetrics();
            this.refreshRunsList(event.run_id);
        }
        if (eventType === "run.failed") {
            if (this.store.run) {
                this.store.run.status = "failed";
            }
            this.setStatus("Failed");
            this.chatPane.addSystemMessage(`Run failed: ${event.payload?.error || "unknown error"}`);
            this.refreshMetrics();
            this.refreshRunsList(event.run_id);
        }
        if (eventType === "run.cancelled") {
            if (this.store.run) {
                this.store.run.status = "cancelled";
            }
            this.setStatus("Cancelled");
            this.chatPane.addSystemMessage("Run cancelled.");
            this.refreshMetrics();
            this.refreshRunsList(event.run_id);
        }
        if (eventType === "run.approval") {
            this.approvalQueueView?.refresh("pending").catch(() => null);
        }

        if (this.store.selectedSeq === null) {
            this.onTraceSelect(event);
        }

        this.inspectorPane.setMetrics(this.panelMetrics());
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
        this.inspectorPane.setMetrics(this.panelMetrics());

        this.setRunMeta(payload.run);
        this.setStatus(payload.run.status || "idle");
        this.upsertRunHistory(payload.run);
        this.renderSideRunSummary(payload.run);

        const selected = this.store.getSelectedEvent();
        if (selected) {
            this.tracePane.selectSeq(selected.seq);
            this.statePane.setSelectedEvent(selected);
            this.inspectorPane.setEvent(selected);
            this.renderSideEventSummary(selected);
        } else if (this.store.events.length > 0) {
            const latestEvent = this.store.events[this.store.events.length - 1];
            this.onTraceSelect(latestEvent);
        } else {
            this.renderSideEventSummary(null);
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
            await this.refreshMetrics();
            await this.refreshRunsList(payload.run.run_id);
        }
    }

    async _activateSpawnedRun(run) {
        if (!run || !run.run_id) {
            return;
        }
        this.store.reset();
        this.setRunMeta(run);
        this.setStatus(run.status || "queued");
        this.upsertRunHistory(run);
        await this.refreshRunDetail(run.run_id);
        this.connectStream(run.run_id);
        this.refreshMetrics();
    }

    async replayFromStep(stepSeq) {
        const runID = this.activeRunID();
        if (!runID) {
            return;
        }
        const normalizedSeq = Number.parseInt(stepSeq, 10);
        const payload = await replayRun(this.apiBase, runID, {
            replay_from_seq: Number.isNaN(normalizedSeq) ? undefined : normalizedSeq,
        });
        if (payload?.run?.run_id) {
            await this._activateSpawnedRun(payload.run);
        } else if (payload?.message) {
            this.setStatus(String(payload.message));
        }
    }

    async replayWithState(stepSeq, stateOverrides) {
        const runID = this.activeRunID();
        if (!runID) {
            return;
        }
        const normalizedSeq = Number.parseInt(stepSeq, 10);
        const payload = await replayRun(this.apiBase, runID, {
            replay_from_seq: Number.isNaN(normalizedSeq) ? undefined : normalizedSeq,
            state_overrides: (stateOverrides && typeof stateOverrides === "object") ? stateOverrides : {},
        });
        if (payload?.run?.run_id) {
            await this._activateSpawnedRun(payload.run);
        } else if (payload?.message) {
            this.setStatus(String(payload.message));
        }
    }

    async replayActiveRun() {
        const runID = this.activeRunID();
        if (!runID) {
            return;
        }
        const selected = this.store.getSelectedEvent();
        const stepSeq = selected?.seq ? Number.parseInt(selected.seq, 10) : undefined;
        await this.replayFromStep(stepSeq);
    }

    async resumeActiveRun() {
        const runID = this.activeRunID();
        if (!runID) {
            return;
        }
        const payload = await resumeRun(this.apiBase, runID);
        if (payload?.run?.run_id) {
            await this._activateSpawnedRun(payload.run);
        }
    }

    async init() {
        this.bindEvents();
        if (window.panelMan && typeof window.panelMan.showPanel === "function" && window.innerWidth > 980) {
            window.panelMan.showPanel("left");
            window.panelMan.showPanel("right");
        }
        await this.loadBootstrap();
        try {
            await this.agentBuilder.init(this.bootstrapData?.agent_definitions || []);
        } catch (error) {
            this.setStatus(`Agent definitions init warning: ${error}`);
        }
        try {
            await this.toolRegistryView.init(
                this.bootstrapData?.tool_registry || [],
                Boolean(this.bootstrapData?.can_write_registry),
            );
        } catch (error) {
            this.setStatus(`Tool registry init warning: ${error}`);
        }
        try {
            await this.sandboxPolicyEditor.init(
                this.bootstrapData?.tool_sandbox_policies || [],
                Boolean(this.bootstrapData?.can_write_registry),
            );
        } catch (error) {
            this.setStatus(`Sandbox policy init warning: ${error}`);
        }
        try {
            await this.approvalQueueView.init(
                this.bootstrapData?.approval_queue || [],
                Boolean(this.bootstrapData?.can_manage_approvals),
            );
        } catch (error) {
            this.setStatus(`Approval queue init warning: ${error}`);
        }
        try {
            await this.toolHarnessView.init(
                this.bootstrapData?.tool_harness_cases || [],
                this.bootstrapData?.tool_registry || [],
                Boolean(this.bootstrapData?.can_write_harness),
            );
        } catch (error) {
            this.setStatus(`Tool harness init warning: ${error}`);
        }
        await this.loadModelInventory();
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
