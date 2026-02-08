class InspectorPane {
    constructor(containerElement) {
        this.container = containerElement;
        this.event = null;
        this.metrics = null;
        this.schemaValidation = null;
        this.render();
    }

    setEvent(event) {
        this.event = event;
        this.render();
    }

    setMetrics(metrics) {
        this.metrics = metrics;
        this.render();
    }

    setSchemaValidation(validation) {
        this.schemaValidation = validation;
        this.render();
    }

    render() {
        if (!this.container) {
            return;
        }

        this.container.innerHTML = "";

        if (!this.event && !this.metrics && !this.schemaValidation) {
            const empty = document.createElement("div");
            empty.className = "ap-empty";
            empty.textContent = "Select a trace event to inspect payloads.";
            this.container.appendChild(empty);
            return;
        }

        if (this.event) {
            const eventNode = document.createElement("pre");
            eventNode.className = "ap-json";
            eventNode.textContent = JSON.stringify(this.event, null, 2);
            this.container.appendChild(eventNode);
        }

        if (this.metrics) {
            const metricsNode = document.createElement("pre");
            metricsNode.className = "ap-json";
            metricsNode.textContent = JSON.stringify({ metrics: this.metrics }, null, 2);
            this.container.appendChild(metricsNode);
        }

        if (this.schemaValidation) {
            const schemaNode = document.createElement("pre");
            schemaNode.className = "ap-json";
            schemaNode.textContent = JSON.stringify({ schema_validation: this.schemaValidation }, null, 2);
            this.container.appendChild(schemaNode);
        }
    }
}

export { InspectorPane };
