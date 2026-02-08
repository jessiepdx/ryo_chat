import { renderVersionHistory } from "./version-history.js";

const defaultDefinition = Object.freeze({
    schema: "ryo.agent_definition.v1",
    identity: {
        name: "New Agent",
        description: "Describe this agent's responsibilities.",
        tags: ["playground"],
        visibility: "private",
    },
    system_prompt: {
        strategy: "policy",
        policy_name: "chat_conversation",
        text: "",
    },
    model_policy: {
        default_model: "",
        allowed_models: [],
        capability_models: {},
        temperature: 0.2,
        top_p: 0.9,
        seed: 42,
    },
    tool_access_policy: {
        enabled_tools: [],
        denied_tools: [],
        per_tool: {},
        custom_tools: [],
    },
    memory_strategy: {
        short_term_strategy: "trim_last_n",
        long_term_strategy: "episodic",
        token_budget: 2048,
        ttl_seconds: 86400,
    },
    guardrail_hooks: {
        pre: [],
        mid: [],
        post: [],
        allow_internal_diagnostics: false,
    },
    orchestration: {
        pattern: "single",
        delegation_enabled: false,
        planner: "",
        executor: "",
        verifier: "",
    },
});

function cloneDefaultDefinition() {
    return JSON.parse(JSON.stringify(defaultDefinition));
}

function asText(value, fallback = "") {
    const text = String(value ?? "").trim();
    return text || fallback;
}

class AgentBuilder {
    constructor(apiBase, refs, callbacks = {}) {
        this.apiBase = `${String(apiBase || "").replace(/\/$/, "")}/agent-definitions`;
        this.refs = refs || {};
        this.callbacks = callbacks || {};
        this.listing = [];
        this.current = null;
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

    _selectedDefinitionID() {
        return asText(this.refs.definitionSelect?.value);
    }

    _selectedVersion() {
        const raw = asText(this.refs.versionSelect?.value);
        if (!raw) {
            return null;
        }
        const parsed = Number.parseInt(raw, 10);
        return Number.isNaN(parsed) ? null : parsed;
    }

    _setDefinitionJSON(payload) {
        if (!this.refs.definitionJSON) {
            return;
        }
        this.refs.definitionJSON.value = JSON.stringify(payload || cloneDefaultDefinition(), null, 2);
    }

    _readDefinitionJSON() {
        const raw = asText(this.refs.definitionJSON?.value);
        if (!raw) {
            return cloneDefaultDefinition();
        }
        try {
            const parsed = JSON.parse(raw);
            if (parsed && typeof parsed === "object") {
                return parsed;
            }
        } catch (_error) {
            throw new Error("Definition editor must contain valid JSON.");
        }
        throw new Error("Definition editor must contain an object payload.");
    }

    _populateDefinitionSelect(definitions) {
        if (!this.refs.definitionSelect) {
            return;
        }
        const previous = this._selectedDefinitionID();
        const items = Array.isArray(definitions) ? definitions : [];
        this.refs.definitionSelect.innerHTML = "";

        const emptyOption = document.createElement("option");
        emptyOption.value = "";
        emptyOption.textContent = "Select definition";
        this.refs.definitionSelect.appendChild(emptyOption);

        for (const item of items) {
            const option = document.createElement("option");
            option.value = asText(item?.definition_id);
            option.textContent = `${asText(item?.name, "Untitled")} (v${Number.parseInt(item?.active_version, 10) || 1})`;
            this.refs.definitionSelect.appendChild(option);
        }

        if (previous && items.some((item) => asText(item?.definition_id) === previous)) {
            this.refs.definitionSelect.value = previous;
        }
    }

    _populateVersionSelect(detail) {
        if (!this.refs.versionSelect) {
            return;
        }
        this.refs.versionSelect.innerHTML = "";
        const versions = Array.isArray(detail?.versions) ? detail.versions : [];
        for (const item of versions.slice().sort((left, right) => Number.parseInt(right?.version, 10) - Number.parseInt(left?.version, 10))) {
            const version = Number.parseInt(item?.version, 10) || 0;
            if (version <= 0) {
                continue;
            }
            const option = document.createElement("option");
            option.value = String(version);
            option.textContent = `v${version}`;
            this.refs.versionSelect.appendChild(option);
        }
        const selected = Number.parseInt(detail?.selected_version, 10) || Number.parseInt(detail?.active_version, 10) || 1;
        this.refs.versionSelect.value = String(selected);
    }

    _applyDetail(detail) {
        this.current = detail || null;
        if (this.refs.nameInput) {
            this.refs.nameInput.value = asText(detail?.name, asText(this.refs.nameInput.value, "New Agent"));
        }
        this._setDefinitionJSON(detail?.definition || cloneDefaultDefinition());
        this._populateVersionSelect(detail || {});
        renderVersionHistory(this.refs.history, detail?.versions || [], detail?.selected_version || detail?.active_version);
    }

    async _request(path = "", options = {}) {
        const response = await fetch(`${this.apiBase}${path}`, options);
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || payload.status !== "ok") {
            throw new Error(asText(payload.message, `Request failed (${response.status})`));
        }
        return payload;
    }

    async refreshList() {
        const payload = await this._request("");
        this.listing = Array.isArray(payload.definitions) ? payload.definitions : [];
        this._populateDefinitionSelect(this.listing);
        return this.listing;
    }

    async loadSelected() {
        const definitionID = this._selectedDefinitionID();
        if (!definitionID) {
            this.current = null;
            this._applyDetail({ definition: cloneDefaultDefinition(), versions: [] });
            return null;
        }
        const version = this._selectedVersion();
        const query = (version && version > 0) ? `?version=${encodeURIComponent(String(version))}` : "";
        const payload = await this._request(`/${encodeURIComponent(definitionID)}${query}`);
        this._applyDetail(payload.definition || {});
        return this.current;
    }

    async createNewDefinition() {
        const payload = {
            definition: this._readDefinitionJSON(),
            change_summary: asText(this.refs.summaryInput?.value, "Initial definition"),
        };
        const response = await this._request("", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        this.status("Created agent definition.", "success");
        await this.refreshList();
        if (this.refs.definitionSelect) {
            this.refs.definitionSelect.value = asText(response.definition?.definition_id);
        }
        await this.loadSelected();
    }

    async saveVersion() {
        const definitionID = this._selectedDefinitionID();
        if (!definitionID) {
            throw new Error("Select an agent definition before saving a new version.");
        }
        const payload = {
            definition: this._readDefinitionJSON(),
            change_summary: asText(this.refs.summaryInput?.value, "Updated definition"),
        };
        await this._request(`/${encodeURIComponent(definitionID)}/versions`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        this.status("Saved new definition version.", "success");
        await this.refreshList();
        await this.loadSelected();
    }

    async rollbackToSelectedVersion() {
        const definitionID = this._selectedDefinitionID();
        const version = this._selectedVersion();
        if (!definitionID || !version) {
            throw new Error("Select a definition and target version to rollback.");
        }
        await this._request(`/${encodeURIComponent(definitionID)}/rollback`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                target_version: version,
                change_summary: asText(this.refs.summaryInput?.value, `Rollback to v${version}`),
            }),
        });
        this.status(`Rolled back using v${version}.`, "success");
        await this.refreshList();
        await this.loadSelected();
    }

    async exportSelected(format = "json") {
        const definitionID = this._selectedDefinitionID();
        if (!definitionID) {
            throw new Error("Select an agent definition to export.");
        }
        const version = this._selectedVersion();
        const query = new URLSearchParams();
        query.set("format", format);
        if (version) {
            query.set("version", String(version));
        }
        const payload = await this._request(`/${encodeURIComponent(definitionID)}/export?${query.toString()}`);
        return asText(payload.payload);
    }

    async importFromEditor(format = "json") {
        const rawPayload = asText(this.refs.definitionJSON?.value);
        if (!rawPayload) {
            throw new Error("Definition editor is empty.");
        }
        const response = await this._request("/import", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                format,
                raw_payload: rawPayload,
                change_summary: asText(this.refs.summaryInput?.value, "Imported definition"),
            }),
        });
        this.status("Imported definition.", "success");
        await this.refreshList();
        if (this.refs.definitionSelect) {
            this.refs.definitionSelect.value = asText(response.definition?.definition_id);
        }
        await this.loadSelected();
    }

    getRunBinding() {
        const definitionID = this._selectedDefinitionID();
        const version = this._selectedVersion();
        const runtimeDefinition = this.current?.definition;
        return {
            definition_id: definitionID || "",
            definition_version: version,
            definition: runtimeDefinition && typeof runtimeDefinition === "object"
                ? JSON.parse(JSON.stringify(runtimeDefinition))
                : null,
        };
    }

    async copyExport(format = "json") {
        const text = await this.exportSelected(format);
        if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
            await navigator.clipboard.writeText(text);
            this.status(`Copied ${format.toUpperCase()} export to clipboard.`, "success");
            return;
        }
        this.status("Clipboard API unavailable. Export payload left in editor.", "warn");
        if (this.refs.definitionJSON) {
            this.refs.definitionJSON.value = text;
        }
    }

    bind() {
        this.refs.refreshBtn?.addEventListener("click", async () => {
            try {
                await this.refreshList();
                await this.loadSelected();
                this.status("Refreshed definitions.");
            } catch (error) {
                this.status(String(error), "error");
            }
        });
        this.refs.definitionSelect?.addEventListener("change", async () => {
            try {
                await this.loadSelected();
            } catch (error) {
                this.status(String(error), "error");
            }
        });
        this.refs.versionSelect?.addEventListener("change", async () => {
            try {
                await this.loadSelected();
            } catch (error) {
                this.status(String(error), "error");
            }
        });
        this.refs.newBtn?.addEventListener("click", async () => {
            try {
                await this.createNewDefinition();
            } catch (error) {
                this.status(String(error), "error");
            }
        });
        this.refs.saveVersionBtn?.addEventListener("click", async () => {
            try {
                await this.saveVersion();
            } catch (error) {
                this.status(String(error), "error");
            }
        });
        this.refs.rollbackBtn?.addEventListener("click", async () => {
            try {
                await this.rollbackToSelectedVersion();
            } catch (error) {
                this.status(String(error), "error");
            }
        });
        this.refs.exportBtn?.addEventListener("click", async () => {
            try {
                await this.copyExport("json");
            } catch (error) {
                this.status(String(error), "error");
            }
        });
        this.refs.importBtn?.addEventListener("click", async () => {
            try {
                await this.importFromEditor("json");
            } catch (error) {
                this.status(String(error), "error");
            }
        });
    }

    async init(bootstrapDefinitions = []) {
        this.bind();
        if (Array.isArray(bootstrapDefinitions) && bootstrapDefinitions.length > 0) {
            this.listing = bootstrapDefinitions.slice();
            this._populateDefinitionSelect(this.listing);
        } else {
            await this.refreshList();
        }
        if (this.listing.length > 0 && this.refs.definitionSelect && !this.refs.definitionSelect.value) {
            this.refs.definitionSelect.value = asText(this.listing[0]?.definition_id);
        }
        await this.loadSelected();
        if (!this.current) {
            this._applyDetail({ definition: cloneDefaultDefinition(), versions: [] });
        }
    }
}

export { AgentBuilder, cloneDefaultDefinition };
