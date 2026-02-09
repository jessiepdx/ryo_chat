function asText(value, fallback = "") {
    const text = String(value ?? "").trim();
    return text || fallback;
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

function asNumber(value, fallback) {
    const parsed = Number.parseFloat(String(value ?? "").trim());
    if (Number.isFinite(parsed)) {
        return parsed;
    }
    return Number.parseFloat(String(fallback ?? 0)) || 0;
}

function splitList(value) {
    return String(value ?? "")
        .split(/[\n,]/g)
        .map((entry) => String(entry || "").trim())
        .filter(Boolean);
}

class SandboxPolicyEditor {
    constructor(apiBase, refs, callbacks = {}) {
        this.apiBase = `${String(apiBase || "").replace(/\/$/, "")}/tools/sandbox-policies`;
        this.refs = refs || {};
        this.callbacks = callbacks || {};
        this.policies = [];
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

    _setWriteState() {
        const disabled = !this.canWrite;
        for (const key of ["saveBtn", "deleteBtn", "newBtn"]) {
            const element = this.refs[key];
            if (element) {
                element.disabled = disabled;
            }
        }
        if (this.refs.writeHint) {
            this.refs.writeHint.textContent = disabled
                ? "Read-only: admin/owner role required to save sandbox policy."
                : "Writable: save per-tool sandbox policy overrides.";
        }
    }

    _policyFromRow(row) {
        const sandbox = row?.sandbox_policy && typeof row.sandbox_policy === "object"
            ? row.sandbox_policy
            : {};
        return {
            tool_name: asText(row?.tool_name || row?.name),
            side_effect_class: asText(row?.side_effect_class || sandbox?.side_effect_class, "read_only"),
            approval_required: asBool(row?.approval_required, asBool(sandbox?.require_approval, false)),
            dry_run: asBool(row?.dry_run, asBool(sandbox?.dry_run, false)),
            approval_timeout_seconds: asNumber(
                sandbox?.approval_timeout_seconds,
                45,
            ),
            execution_timeout_ceiling: asNumber(
                sandbox?.execution_timeout_ceiling,
                30,
            ),
            max_memory_mb: Math.max(16, Number.parseInt(sandbox?.max_memory_mb, 10) || 512),
            network_enabled: asBool(sandbox?.network?.enabled, true),
            network_allowlist_domains: Array.isArray(sandbox?.network?.allowlist_domains)
                ? sandbox.network.allowlist_domains.map((item) => asText(item)).filter(Boolean)
                : [],
            filesystem_mode: asText(sandbox?.filesystem?.mode, "none"),
            filesystem_allowed_paths: Array.isArray(sandbox?.filesystem?.allowed_paths)
                ? sandbox.filesystem.allowed_paths.map((item) => asText(item)).filter(Boolean)
                : [],
            source: asText(row?.source, "builtin"),
        };
    }

    _applyCurrent(row) {
        this.current = this._policyFromRow(row || {});
        if (this.refs.toolSelect) {
            const optionValue = asText(this.current.tool_name);
            if (optionValue) {
                this.refs.toolSelect.value = optionValue;
            }
        }
        if (this.refs.toolNameInput) {
            this.refs.toolNameInput.value = asText(this.current.tool_name);
        }
        if (this.refs.sourceLabel) {
            this.refs.sourceLabel.textContent = asText(this.current.source, "builtin");
        }
        if (this.refs.sideEffectClass) {
            this.refs.sideEffectClass.value = asText(this.current.side_effect_class, "read_only");
        }
        if (this.refs.approvalRequired) {
            this.refs.approvalRequired.checked = asBool(this.current.approval_required, false);
        }
        if (this.refs.dryRunEnabled) {
            this.refs.dryRunEnabled.checked = asBool(this.current.dry_run, false);
        }
        if (this.refs.approvalTimeoutSeconds) {
            this.refs.approvalTimeoutSeconds.value = String(this.current.approval_timeout_seconds);
        }
        if (this.refs.timeoutCeilingSeconds) {
            this.refs.timeoutCeilingSeconds.value = String(this.current.execution_timeout_ceiling);
        }
        if (this.refs.maxMemoryMB) {
            this.refs.maxMemoryMB.value = String(this.current.max_memory_mb);
        }
        if (this.refs.networkEnabled) {
            this.refs.networkEnabled.checked = asBool(this.current.network_enabled, true);
        }
        if (this.refs.networkAllowlistDomains) {
            this.refs.networkAllowlistDomains.value = (this.current.network_allowlist_domains || []).join(", ");
        }
        if (this.refs.filesystemMode) {
            this.refs.filesystemMode.value = asText(this.current.filesystem_mode, "none");
        }
        if (this.refs.filesystemAllowedPaths) {
            this.refs.filesystemAllowedPaths.value = (this.current.filesystem_allowed_paths || []).join("\n");
        }
    }

    _populateSelect() {
        if (!this.refs.toolSelect) {
            return;
        }
        const previous = asText(this.refs.toolSelect.value);
        this.refs.toolSelect.innerHTML = "";
        const empty = document.createElement("option");
        empty.value = "";
        empty.textContent = "Select tool";
        this.refs.toolSelect.appendChild(empty);
        for (const row of this.policies) {
            const toolName = asText(row?.tool_name || row?.name);
            if (!toolName) {
                continue;
            }
            const option = document.createElement("option");
            option.value = toolName;
            const source = asText(row?.source, "builtin");
            option.textContent = `${toolName} (${source})`;
            this.refs.toolSelect.appendChild(option);
        }
        if (previous && this.policies.some((row) => asText(row?.tool_name || row?.name) === previous)) {
            this.refs.toolSelect.value = previous;
        }
    }

    _collectPayload() {
        const toolName = asText(this.refs.toolNameInput?.value || this.refs.toolSelect?.value);
        if (!toolName) {
            throw new Error("Tool name is required.");
        }
        const sideEffect = asText(this.refs.sideEffectClass?.value, "read_only");
        return {
            tool_name: toolName,
            side_effect_class: sideEffect,
            approval_required: asBool(this.refs.approvalRequired?.checked, false),
            dry_run: asBool(this.refs.dryRunEnabled?.checked, false),
            sandbox_policy: {
                tool_name: toolName,
                side_effect_class: sideEffect,
                require_approval: asBool(this.refs.approvalRequired?.checked, false),
                dry_run: asBool(this.refs.dryRunEnabled?.checked, false),
                approval_timeout_seconds: asNumber(this.refs.approvalTimeoutSeconds?.value, 45),
                execution_timeout_ceiling: asNumber(this.refs.timeoutCeilingSeconds?.value, 30),
                max_memory_mb: Math.max(16, Number.parseInt(this.refs.maxMemoryMB?.value, 10) || 512),
                network: {
                    enabled: asBool(this.refs.networkEnabled?.checked, true),
                    allowlist_domains: splitList(this.refs.networkAllowlistDomains?.value),
                },
                filesystem: {
                    mode: asText(this.refs.filesystemMode?.value, "none"),
                    allowed_paths: splitList(this.refs.filesystemAllowedPaths?.value),
                },
            },
        };
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
        this.policies = Array.isArray(payload.policies) ? payload.policies : [];
        this.canWrite = asBool(payload.can_write, false);
        this._setWriteState();
        this._populateSelect();
    }

    loadSelected() {
        const selected = asText(this.refs.toolSelect?.value);
        if (!selected) {
            this._applyCurrent({});
            return;
        }
        const row = this.policies.find((item) => asText(item?.tool_name || item?.name) === selected);
        this._applyCurrent(row || {});
    }

    async save() {
        if (!this.canWrite) {
            throw new Error("Sandbox policy is read-only for this account.");
        }
        const payload = { policy: this._collectPayload() };
        await this._request("", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        this.status("Sandbox policy saved.", "success");
        await this.refresh();
        const selected = asText(payload?.policy?.tool_name);
        if (this.refs.toolSelect && selected) {
            this.refs.toolSelect.value = selected;
        }
        this.loadSelected();
    }

    async removeSelected() {
        if (!this.canWrite) {
            throw new Error("Sandbox policy is read-only for this account.");
        }
        const selected = asText(this.refs.toolSelect?.value || this.refs.toolNameInput?.value);
        if (!selected) {
            throw new Error("Select a policy to remove.");
        }
        await this._request(`/${encodeURIComponent(selected)}`, { method: "DELETE" });
        this.status("Sandbox policy removed.", "success");
        await this.refresh();
        if (this.refs.toolSelect) {
            this.refs.toolSelect.value = "";
        }
        this.loadSelected();
    }

    bind() {
        this.refs.refreshBtn?.addEventListener("click", async () => {
            try {
                await this.refresh();
                this.loadSelected();
                this.status("Sandbox policies refreshed.");
            } catch (error) {
                this.status(String(error), "error");
            }
        });
        this.refs.toolSelect?.addEventListener("change", () => this.loadSelected());
        this.refs.newBtn?.addEventListener("click", () => {
            if (this.refs.toolSelect) {
                this.refs.toolSelect.value = "";
            }
            this._applyCurrent({});
            this.status("Editing new sandbox policy.");
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
    }

    async init(bootstrapPolicies = [], canWriteRegistry = false) {
        this.bind();
        this.canWrite = Boolean(canWriteRegistry);
        if (Array.isArray(bootstrapPolicies) && bootstrapPolicies.length > 0) {
            this.policies = bootstrapPolicies.slice();
            this._setWriteState();
            this._populateSelect();
        } else {
            await this.refresh();
        }
        this.loadSelected();
    }
}

export { SandboxPolicyEditor };
