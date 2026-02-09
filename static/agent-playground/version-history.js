function _asText(value, fallback = "") {
    const text = String(value ?? "").trim();
    return text || fallback;
}

function renderVersionHistory(containerElement, versions, activeVersion) {
    if (!containerElement) {
        return;
    }

    const list = Array.isArray(versions) ? versions.slice() : [];
    containerElement.innerHTML = "";
    if (list.length === 0) {
        containerElement.textContent = "No version history yet.";
        return;
    }

    const ordered = list
        .map((item) => ({
            version: Number.parseInt(item?.version, 10) || 0,
            created_at: _asText(item?.created_at, "-"),
            author_member_id: Number.parseInt(item?.author_member_id, 10) || 0,
            change_summary: _asText(item?.change_summary, ""),
        }))
        .sort((left, right) => right.version - left.version);

    for (const item of ordered) {
        const row = document.createElement("div");
        row.className = "ap-side-history-row";
        if (Number.parseInt(activeVersion, 10) === item.version) {
            row.classList.add("is-active");
        }

        const title = document.createElement("strong");
        title.textContent = `v${item.version}`;
        row.appendChild(title);

        const meta = document.createElement("span");
        meta.textContent = `${item.created_at} · author ${item.author_member_id || "-"}`;
        row.appendChild(meta);

        if (item.change_summary) {
            const summary = document.createElement("em");
            summary.textContent = item.change_summary;
            row.appendChild(summary);
        }
        containerElement.appendChild(row);
    }
}

export { renderVersionHistory };
