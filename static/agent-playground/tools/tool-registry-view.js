import { ToolFormRenderer, coerceObjectSchema } from "./tool-form-renderer.js";

function asText(value, fallback = "") {
    const text = String(value ?? "").trim();
    return text || fallback;
}

class ToolRegistryView {
    constructor(apiBase, refs, callbacks = {}) {
        this.apiBase = `${String(apiBase || "").replace(/\/$/, "")}/tools`;
        this.refs = refs || {};
        this.callbacks = callbacks || {};
        this.tools = [];
        this.canWrite = false;
        this.current = null;
        this.formRenderer = new ToolFormRenderer(this.refs.metadataForm, this.refs.argPreviewForm);
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

    _setWriteState() {
        const disabled = !this.canWrite;
        for (const refName of ["saveBtn", "deleteBtn", "newBtn"]) {
            const node = this.refs[refName];
            if (node) {
                node.disabled = disabled;
            }
        }
        if (this.refs.writeHint) {
            this.refs.writeHint.textContent = disabled
                ? "Read-only: admin/owner role required for registry writes."
                : "Writable: custom tools can be created/updated here.";
        }
    }

    _setSchemaText(schemaObject) {
        if (!this.refs.schemaJSON) {
            return;
        }
        this.refs.schemaJSON.value = JSON.stringify(coerceObjectSchema(schemaObject), null, 2);
    }

    _readSchemaText() {
        const raw = asText(this.refs.schemaJSON?.value);
        if (!raw) {
            return { type: "object", properties: {}, required: [] };
        }
        let parsed;
        try {
            parsed = JSON.parse(raw);
        } catch (_error) {
            throw new Error("Tool input schema must be valid JSON.");
        }
        return coerceObjectSchema(parsed);
    }

    _readStaticResult() {
        const raw = asText(this.refs.staticResultJSON?.value);
        if (!raw) {
            return null;
        }
        try {
            return JSON.parse(raw);
        } catch (_error) {
            throw new Error("Static result payload must be valid JSON.");
        }
    }

    _applyCurrent(tool) {
        this.current = tool || null;
        this.formRenderer.renderMetadata({
            name: asText(tool?.name),
            description: asText(tool?.description),
            enabled: tool?.enabled !== false,
            auth_requirements: asText(tool?.auth_requirements),
            side_effect_class: asText(tool?.side_effect_class, "read_only"),
            approval_required: tool?.approval_required === true,
            dry_run: tool?.dry_run === true,
            approval_timeout_seconds: Number.parseFloat(tool?.sandbox_policy?.approval_timeout_seconds) || 45,
            rate_limit_per_minute: Number.parseInt(tool?.rate_limit_per_minute, 10) || 0,
            handler_mode: asText(tool?.handler_mode, "echo"),
            required_api_key: asText(tool?.required_api_key),
            timeout_seconds: Number.parseFloat(tool?.timeout_seconds) || 8,
            max_retries: Number.parseInt(tool?.max_retries, 10) || 0,
        });
        this._setSchemaText(tool?.input_schema || { type: "object", properties: {}, required: [] });
        this.formRenderer.renderArgumentPreview(tool?.input_schema || { type: "object", properties: {} }, {});
        if (this.refs.staticResultJSON) {
            this.refs.staticResultJSON.value = tool?.handler_mode === "static"
                ? JSON.stringify(tool?.static_result ?? { status: "success" }, null, 2)
                : "";
        }
        if (this.refs.selectedSummary) {
            const source = asText(tool?.source, "builtin");
            const enabled = tool?.enabled === false ? "disabled" : "enabled";
            this.refs.selectedSummary.textContent = `${source} · ${enabled}`;
        }
    }

    _populateTools() {
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
            const option = document.createElement("option");
            option.value = asText(tool?.name);
            const source = asText(tool?.source, "builtin");
            option.textContent = `${asText(tool?.name)} (${source})`;
            this.refs.toolSelect.appendChild(option);
        }
        if (previous && this.tools.some((tool) => asText(tool?.name) === previous)) {
            this.refs.toolSelect.value = previous;
        }
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
        const payload = await this._request("");
        this.tools = Array.isArray(payload.tools) ? payload.tools : [];
        this.canWrite = Boolean(payload.can_write);
        this._setWriteState();
        this._populateTools();
    }

    loadSelected() {
        const name = asText(this.refs.toolSelect?.value);
        if (!name) {
            this._applyCurrent({
                source: "custom",
                enabled: true,
                name: "",
                description: "",
                input_schema: { type: "object", properties: {}, required: [] },
                handler_mode: "echo",
                static_result: null,
            });
            return;
        }
        const selected = this.tools.find((tool) => asText(tool?.name) === name);
        this._applyCurrent(selected || null);
    }

    async save() {
        if (!this.canWrite) {
            throw new Error("Tool registry is read-only for this account.");
        }
        const metadata = this.formRenderer.metadataValues();
        if (!metadata) {
            throw new Error("Metadata form is invalid.");
        }
        const inputSchema = this._readSchemaText();
        const payload = {
            tool: {
                ...metadata,
                input_schema: inputSchema,
                static_result: this._readStaticResult(),
                sandbox_policy: {
                    tool_name: asText(metadata.name),
                    side_effect_class: asText(metadata.side_effect_class, "read_only"),
                    require_approval: metadata.approval_required === true,
                    dry_run: metadata.dry_run === true,
                    approval_timeout_seconds: Number.parseFloat(metadata.approval_timeout_seconds) || 45,
                },
            },
        };

        await this._request("", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        this.status("Tool registry updated.", "success");
        await this.refresh();
        const toolName = asText(metadata.name);
        if (this.refs.toolSelect && toolName) {
            this.refs.toolSelect.value = toolName;
        }
        this.loadSelected();
    }

    async removeSelected() {
        if (!this.canWrite) {
            throw new Error("Tool registry is read-only for this account.");
        }
        const name = asText(this.refs.toolSelect?.value);
        if (!name) {
            throw new Error("Select a custom tool to remove.");
        }
        await this._request(`/${encodeURIComponent(name)}`, {
            method: "DELETE",
        });
        this.status("Tool removed from custom registry.", "success");
        await this.refresh();
        this.refs.toolSelect.value = "";
        this.loadSelected();
    }

    bind() {
        this.refs.refreshBtn?.addEventListener("click", async () => {
            try {
                await this.refresh();
                this.loadSelected();
                this.status("Tool registry refreshed.");
            } catch (error) {
                this.status(String(error), "error");
            }
        });
        this.refs.toolSelect?.addEventListener("change", () => {
            this.loadSelected();
        });
        this.refs.newBtn?.addEventListener("click", () => {
            this.refs.toolSelect.value = "";
            this.loadSelected();
            this.status("Editing new custom tool payload.");
        });
        this.refs.saveBtn?.addEventListener("click", async () => {
            try {
                await this.save();
            } catch (error) {
                this.status(String(error), "error");
            }
        });
        this.refs.deleteBtn?.addEventListener("click", async () => {
            try {
                await this.removeSelected();
            } catch (error) {
                this.status(String(error), "error");
            }
        });
        this.refs.schemaJSON?.addEventListener("input", () => {
            try {
                const schema = this._readSchemaText();
                this.formRenderer.renderArgumentPreview(schema, {});
            } catch (_error) {
                return;
            }
        });
    }

    async init(bootstrapTools = [], canWriteRegistry = false) {
        this.bind();
        this.canWrite = Boolean(canWriteRegistry);
        if (Array.isArray(bootstrapTools) && bootstrapTools.length > 0) {
            this.tools = bootstrapTools.slice();
            this._setWriteState();
            this._populateTools();
        } else {
            await this.refresh();
        }
        this.loadSelected();
    }
}

export { ToolRegistryView };
