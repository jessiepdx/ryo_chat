import { formatHarnessReport } from "./regression-report.js";

function asText(value, fallback = "") {
    const text = String(value ?? "").trim();
    return text || fallback;
}

function asObject(value) {
    return value && typeof value === "object" ? value : {};
}

function asBool(value, fallback = false) {
    if (typeof value === "boolean") {
        return value;
    }
    if (typeof value === "string") {
        const normalized = value.trim().toLowerCase();
        if (["1", "true", "yes", "on"].includes(normalized)) {
            return true;
        }
        if (["0", "false", "no", "off"].includes(normalized)) {
            return false;
        }
    }
    return Boolean(fallback);
}

class ToolHarnessView {
    constructor(apiBase, refs, callbacks = {}) {
        this.apiBase = `${String(apiBase || "").replace(/\/$/, "")}/tools/harness`;
        this.refs = refs || {};
        this.callbacks = callbacks || {};
        this.cases = [];
        this.tools = [];
        this.current = null;
        this.canWrite = false;
    }

    status(message, kind = "info") {
        if (this.refs.status) {
            this.refs.status.textContent = asText(message, "Idle");
            this.refs.status.dataset.kind = asText(kind, "info");
        }
        if (typeof this.callbacks.onStatus === "function") {
            this.callbacks.onStatus(asText(message, "Idle"));
        }
    }

    _renderReport(report) {
        if (!this.refs.report) {
            return;
        }
        this.refs.report.textContent = formatHarnessReport(asObject(report));
    }

    _setWriteState() {
        const disabled = !this.canWrite;
        for (const key of ["newBtn", "saveBtn", "runBtn", "saveGoldenBtn", "saveContractBtn", "deleteBtn"]) {
            const node = this.refs[key];
            if (node) {
                node.disabled = disabled;
            }
        }
        if (this.refs.writeHint) {
            this.refs.writeHint.textContent = disabled
                ? "Read-only: admin/owner role required for harness edits."
                : "Writable: create fixtures, run tools, and store baselines.";
        }
    }

    _populateToolSelect() {
        if (!this.refs.toolSelect) {
            return;
        }
        const previous = asText(this.refs.toolSelect.value);
        this.refs.toolSelect.innerHTML = "";
        const empty = document.createElement("option");
        empty.value = "";
        empty.textContent = "Select tool";
        this.refs.toolSelect.appendChild(empty);
        for (const tool of this.tools) {
            const row = asObject(tool);
            const toolName = asText(row.name);
            if (!toolName) {
                continue;
            }
            const option = document.createElement("option");
            option.value = toolName;
            option.textContent = `${toolName} (${asText(row.source, "builtin")})`;
            this.refs.toolSelect.appendChild(option);
        }
        if (previous) {
            this.refs.toolSelect.value = previous;
        }
    }

    _populateCaseSelect() {
        if (!this.refs.caseSelect) {
            return;
        }
        const previous = asText(this.refs.caseSelect.value);
        this.refs.caseSelect.innerHTML = "";
        const empty = document.createElement("option");
        empty.value = "";
        empty.textContent = "Select case";
        this.refs.caseSelect.appendChild(empty);
        for (const item of this.cases) {
            const row = asObject(item);
            const caseID = asText(row.case_id);
            if (!caseID) {
                continue;
            }
            const fixtureName = asText(row.fixture_name, caseID);
            const toolName = asText(row.tool_name, "tool");
            const mode = asText(row.execution_mode, "real");
            const option = document.createElement("option");
            option.value = caseID;
            option.textContent = `${fixtureName} · ${toolName} · ${mode}`;
            this.refs.caseSelect.appendChild(option);
        }
        if (previous) {
            this.refs.caseSelect.value = previous;
        }
    }

    _readInputArgs() {
        const raw = asText(this.refs.inputJSON?.value);
        if (!raw) {
            return {};
        }
        let parsed;
        try {
            parsed = JSON.parse(raw);
        } catch (_error) {
            throw new Error("Fixture input must be valid JSON.");
        }
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
            throw new Error("Fixture input must be a JSON object.");
        }
        return parsed;
    }

    _collectCasePayload() {
        const caseID = asText(this.refs.caseSelect?.value);
        const toolName = asText(this.refs.toolSelect?.value);
        if (!toolName) {
            throw new Error("Select a tool.");
        }
        const fixtureName = asText(this.refs.fixtureName?.value, `${toolName} fixture`);
        const executionMode = asText(this.refs.mode?.value, "real").toLowerCase();
        if (!["real", "mock"].includes(executionMode)) {
            throw new Error("Execution mode must be real or mock.");
        }
        return {
            case_id: caseID || undefined,
            tool_name: toolName,
            fixture_name: fixtureName,
            description: asText(this.refs.description?.value),
            execution_mode: executionMode,
            input_args: this._readInputArgs(),
            enabled: true,
        };
    }

    _applyCase(row) {
        const item = asObject(row);
        this.current = Object.keys(item).length > 0 ? item : null;

        if (this.refs.caseSelect) {
            const caseID = asText(item.case_id);
            if (caseID) {
                this.refs.caseSelect.value = caseID;
            }
        }
        if (this.refs.toolSelect) {
            this.refs.toolSelect.value = asText(item.tool_name);
        }
        if (this.refs.fixtureName) {
            this.refs.fixtureName.value = asText(item.fixture_name);
        }
        if (this.refs.description) {
            this.refs.description.value = asText(item.description);
        }
        if (this.refs.mode) {
            this.refs.mode.value = asText(item.execution_mode, "real");
        }
        if (this.refs.inputJSON) {
            this.refs.inputJSON.value = JSON.stringify(asObject(item.input_args), null, 2);
        }
        this._renderReport(item.last_report || {});
    }

    _selectCaseByID(caseID) {
        const cleanID = asText(caseID);
        if (!cleanID) {
            this._applyCase({});
            return;
        }
        const selected = this.cases.find((item) => asText(item?.case_id) === cleanID);
        this._applyCase(selected || {});
    }

    async _request(path = "", options = {}) {
        const response = await fetch(`${this.apiBase}${path}`, options);
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || payload.status !== "ok") {
            throw new Error(asText(payload.message, `Request failed (${response.status})`));
        }
        return payload;
    }

    async refresh() {
        const payload = await this._request("/cases");
        this.cases = Array.isArray(payload.cases) ? payload.cases : [];
        this.tools = Array.isArray(payload.tools) ? payload.tools.filter((tool) => tool?.enabled !== false) : [];
        this.canWrite = asBool(payload.can_write, false);
        this._setWriteState();
        this._populateToolSelect();
        this._populateCaseSelect();
    }

    newCase() {
        if (this.refs.caseSelect) {
            this.refs.caseSelect.value = "";
        }
        this._applyCase({});
        if (this.refs.toolSelect) {
            this.refs.toolSelect.value = "";
        }
        if (this.refs.inputJSON) {
            this.refs.inputJSON.value = "{}";
        }
        this._renderReport({});
        this.status("Editing new tool harness case.");
    }

    async saveCase() {
        if (!this.canWrite) {
            throw new Error("Tool harness is read-only for this account.");
        }
        const payload = {
            case: this._collectCasePayload(),
        };
        const response = await this._request("/cases", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        this.cases = Array.isArray(response.cases) ? response.cases : this.cases;
        this._populateCaseSelect();
        const caseID = asText(response?.case?.case_id);
        if (this.refs.caseSelect && caseID) {
            this.refs.caseSelect.value = caseID;
        }
        this._selectCaseByID(caseID);
        this.status("Tool harness case saved.", "success");
    }

    async runCase() {
        if (!this.canWrite) {
            throw new Error("Tool harness run is read-only for this account.");
        }
        const caseID = asText(this.refs.caseSelect?.value);
        if (!caseID) {
            throw new Error("Save or select a harness case before running.");
        }
        const executionMode = asText(this.refs.mode?.value, "real");
        const payload = await this._request("/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                case_id: caseID,
                execution_mode: executionMode,
                persist_run: true,
            }),
        });
        if (payload.case && typeof payload.case === "object") {
            const index = this.cases.findIndex((item) => asText(item?.case_id) === caseID);
            if (index >= 0) {
                this.cases[index] = payload.case;
            } else {
                this.cases.unshift(payload.case);
            }
        }
        this._populateCaseSelect();
        this._selectCaseByID(caseID);
        this._renderReport(payload.report || {});
        this.status("Tool harness run complete.", "success");
    }

    async saveGolden() {
        if (!this.canWrite) {
            throw new Error("Tool harness is read-only for this account.");
        }
        const caseID = asText(this.refs.caseSelect?.value);
        if (!caseID) {
            throw new Error("Select a harness case first.");
        }
        const payload = await this._request(`/cases/${encodeURIComponent(caseID)}/golden`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
        });
        const updated = asObject(payload.case);
        const index = this.cases.findIndex((item) => asText(item?.case_id) === caseID);
        if (index >= 0) {
            this.cases[index] = updated;
        }
        this._selectCaseByID(caseID);
        this.status("Golden output saved.", "success");
    }

    async saveContract() {
        if (!this.canWrite) {
            throw new Error("Tool harness is read-only for this account.");
        }
        const caseID = asText(this.refs.caseSelect?.value);
        if (!caseID) {
            throw new Error("Select a harness case first.");
        }
        const payload = await this._request(`/cases/${encodeURIComponent(caseID)}/contract`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
        });
        const updated = asObject(payload.case);
        const index = this.cases.findIndex((item) => asText(item?.case_id) === caseID);
        if (index >= 0) {
            this.cases[index] = updated;
        }
        this._selectCaseByID(caseID);
        this.status("Contract baseline saved.", "success");
    }

    async deleteCase() {
        if (!this.canWrite) {
            throw new Error("Tool harness is read-only for this account.");
        }
        const caseID = asText(this.refs.caseSelect?.value);
        if (!caseID) {
            throw new Error("Select a case to delete.");
        }
        const payload = await this._request(`/cases/${encodeURIComponent(caseID)}`, {
            method: "DELETE",
        });
        this.cases = Array.isArray(payload.cases) ? payload.cases : [];
        this._populateCaseSelect();
        this.newCase();
        this.status("Tool harness case removed.", "success");
    }

    bind() {
        this.refs.refreshBtn?.addEventListener("click", async () => {
            try {
                const selected = asText(this.refs.caseSelect?.value);
                await this.refresh();
                this._selectCaseByID(selected);
                this.status("Tool harness refreshed.");
            } catch (error) {
                this.status(String(error), "error");
            }
        });
        this.refs.caseSelect?.addEventListener("change", () => {
            this._selectCaseByID(this.refs.caseSelect?.value);
        });
        this.refs.newBtn?.addEventListener("click", () => {
            this.newCase();
        });
        this.refs.saveBtn?.addEventListener("click", async () => {
            try {
                await this.saveCase();
            } catch (error) {
                this.status(String(error), "error");
            }
        });
        this.refs.runBtn?.addEventListener("click", async () => {
            try {
                await this.runCase();
            } catch (error) {
                this.status(String(error), "error");
            }
        });
        this.refs.saveGoldenBtn?.addEventListener("click", async () => {
            try {
                await this.saveGolden();
            } catch (error) {
                this.status(String(error), "error");
            }
        });
        this.refs.saveContractBtn?.addEventListener("click", async () => {
            try {
                await this.saveContract();
            } catch (error) {
                this.status(String(error), "error");
            }
        });
        this.refs.deleteBtn?.addEventListener("click", async () => {
            try {
                await this.deleteCase();
            } catch (error) {
                this.status(String(error), "error");
            }
        });
    }

    async init(bootstrapCases = [], bootstrapTools = [], canWrite = false) {
        this.bind();
        this.canWrite = Boolean(canWrite);
        if (Array.isArray(bootstrapCases) && bootstrapCases.length > 0) {
            this.cases = bootstrapCases.slice();
        }
        if (Array.isArray(bootstrapTools) && bootstrapTools.length > 0) {
            this.tools = bootstrapTools.filter((item) => item?.enabled !== false);
        }
        this._setWriteState();
        this._populateToolSelect();
        this._populateCaseSelect();
        if (this.cases.length === 0 || this.tools.length === 0) {
            await this.refresh();
        }
        const selected = asText(this.refs.caseSelect?.value);
        this._selectCaseByID(selected);
        if (!selected) {
            this.newCase();
        }
    }
}

export { ToolHarnessView };
