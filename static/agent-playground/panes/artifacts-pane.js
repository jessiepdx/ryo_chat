class ArtifactsPane {
    constructor(containerElement) {
        this.container = containerElement;
        this.artifacts = [];
        this.render();
    }

    setArtifacts(artifacts) {
        this.artifacts = Array.isArray(artifacts) ? artifacts.slice() : [];
        this.render();
    }

    render() {
        if (!this.container) {
            return;
        }

        this.container.innerHTML = "";
        if (this.artifacts.length === 0) {
            const empty = document.createElement("div");
            empty.className = "ap-empty";
            empty.textContent = "No artifacts yet.";
            this.container.appendChild(empty);
            return;
        }

        for (const artifact of this.artifacts.slice().reverse()) {
            const row = document.createElement("div");
            row.className = "ap-trace-row";

            const title = document.createElement("div");
            title.textContent = `${artifact.artifact_type} / ${artifact.artifact_name}`;
            row.appendChild(title);

            const meta = document.createElement("div");
            meta.className = "ap-row-meta";
            meta.textContent = artifact.timestamp || "";
            row.appendChild(meta);

            const body = document.createElement("pre");
            body.className = "ap-json";
            body.textContent = JSON.stringify(artifact.artifact || {}, null, 2);
            row.appendChild(body);

            this.container.appendChild(row);
        }
    }
}

export { ArtifactsPane };
