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

function shortText(value, maxLength = 120) {
    const text = asText(value);
    if (!text) {
        return "-";
    }
    if (text.length <= maxLength) {
        return text;
    }
    return `${text.slice(0, Math.max(0, maxLength - 3))}...`;
}

class ApprovalQueueView {
    constructor(apiBase, refs, callbacks = {}) {
        this.apiBase = `${String(apiBase || "").replace(/\/$/, "")}/tool-approvals`;
        this.refs = refs || {};
        this.callbacks = callbacks || {};
        this.approvals = [];
        this.canDecide = false;
        this._pollTimer = null;
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

    _render() {
        if (!this.refs.list) {
            return;
        }
        this.refs.list.innerHTML = "";
        if (!Array.isArray(this.approvals) || this.approvals.length === 0) {
            const empty = document.createElement("div");
            empty.className = "ap-approval-row ap-approval-row-empty";
            empty.textContent = "No pending approvals.";
            this.refs.list.appendChild(empty);
            return;
        }

        for (const approval of this.approvals) {
            const row = document.createElement("div");
            row.className = "ap-approval-row";

            const meta = document.createElement("div");
            meta.className = "ap-approval-meta";
            meta.innerHTML = `
                <strong>${asText(approval?.tool_name, "unknown tool")}</strong>
                <span>Request: ${asText(approval?.request_id, "-")}</span>
                <span>Run: ${asText(approval?.run_id, "-")}</span>
                <span>Status: ${asText(approval?.status, "pending")}</span>
                <span>Reason: ${shortText(approval?.reason, 90)}</span>
                <span>Expires: ${asText(approval?.expires_at, "-")}</span>
            `;
            row.appendChild(meta);

            const actions = document.createElement("div");
            actions.className = "ap-approval-actions";
            const reasonInput = document.createElement("input");
            reasonInput.type = "text";
            reasonInput.placeholder = "Optional decision note";
            reasonInput.className = "ap-approval-reason";
            actions.appendChild(reasonInput);

            const approveBtn = document.createElement("button");
            approveBtn.type = "button";
            approveBtn.textContent = "Approve";
            approveBtn.disabled = !this.canDecide;
            approveBtn.addEventListener("click", async () => {
                try {
                    await this.decide(asText(approval?.request_id), "approve", asText(reasonInput.value));
                } catch (error) {
                    this.status(String(error), "error");
                }
            });
            actions.appendChild(approveBtn);

            const denyBtn = document.createElement("button");
            denyBtn.type = "button";
            denyBtn.textContent = "Deny";
            denyBtn.disabled = !this.canDecide;
            denyBtn.addEventListener("click", async () => {
                try {
                    await this.decide(asText(approval?.request_id), "deny", asText(reasonInput.value));
                } catch (error) {
                    this.status(String(error), "error");
                }
            });
            actions.appendChild(denyBtn);

            row.appendChild(actions);
            this.refs.list.appendChild(row);
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

    async refresh(status = "pending") {
        const query = new URLSearchParams();
        if (status) {
            query.set("status", status);
        }
        query.set("limit", "100");
        const payload = await this._request(`?${query.toString()}`);
        this.approvals = Array.isArray(payload.approvals) ? payload.approvals : [];
        this.canDecide = asBool(payload.can_decide, this.canDecide);
        if (this.refs.writeHint) {
            this.refs.writeHint.textContent = this.canDecide
                ? "Approval decisions enabled."
                : "Read-only approval queue.";
        }
        this._render();
    }

    async decide(requestID, decision, reason = "") {
        const cleanID = asText(requestID);
        if (!cleanID) {
            throw new Error("Approval request id is required.");
        }
        const payload = await this._request(`/${encodeURIComponent(cleanID)}/decision`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ decision, reason }),
        });
        this.status(
            `Approval ${asText(payload?.approval?.status, "updated")} for ${asText(payload?.approval?.tool_name)}.`,
            "success",
        );
        await this.refresh("pending");
    }

    _startPolling() {
        this._stopPolling();
        this._pollTimer = window.setInterval(async () => {
            try {
                await this.refresh("pending");
            } catch (_error) {
                return;
            }
        }, 5000);
    }

    _stopPolling() {
        if (this._pollTimer) {
            window.clearInterval(this._pollTimer);
            this._pollTimer = null;
        }
    }

    bind() {
        this.refs.refreshBtn?.addEventListener("click", async () => {
            try {
                await this.refresh("pending");
                this.status("Approval queue refreshed.");
            } catch (error) {
                this.status(String(error), "error");
            }
        });
    }

    async init(bootstrapApprovals = [], canDecide = false) {
        this.bind();
        this.canDecide = Boolean(canDecide);
        if (Array.isArray(bootstrapApprovals) && bootstrapApprovals.length > 0) {
            this.approvals = bootstrapApprovals.slice();
            this._render();
        } else {
            await this.refresh("pending");
        }
        this._startPolling();
    }
}

export { ApprovalQueueView };
