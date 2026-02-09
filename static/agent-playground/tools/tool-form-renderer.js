import { SchemaFormRenderer } from "../schema-renderer/renderer.js";

const metadataSchema = Object.freeze({
    type: "object",
    required: ["name", "description"],
    properties: {
        name: {
            type: "string",
            description: "Unique tool name (letters, numbers, underscore).",
        },
        description: {
            type: "string",
            description: "Tool description shown to model and users.",
        },
        enabled: {
            type: "boolean",
            default: true,
            description: "Whether this custom tool should be available to runtime.",
        },
        auth_requirements: {
            type: "string",
            description: "Short note describing auth prerequisites.",
        },
        side_effect_class: {
            type: "string",
            enum: ["read_only", "mutating", "sensitive"],
            default: "read_only",
        },
        approval_required: {
            type: "boolean",
            default: false,
        },
        dry_run: {
            type: "boolean",
            default: false,
        },
        approval_timeout_seconds: {
            type: "number",
            minimum: 1,
            default: 45,
        },
        rate_limit_per_minute: {
            type: "integer",
            minimum: 0,
            default: 0,
        },
        handler_mode: {
            type: "string",
            enum: ["echo", "static"],
            default: "echo",
        },
        required_api_key: {
            type: "string",
            description: "Optional key name required to execute this tool.",
        },
        timeout_seconds: {
            type: "number",
            minimum: 0.1,
            default: 8,
        },
        max_retries: {
            type: "integer",
            minimum: 0,
            default: 0,
        },
    },
});

function coerceObjectSchema(inputSchema) {
    const schema = (inputSchema && typeof inputSchema === "object") ? { ...inputSchema } : {};
    if (schema.type !== "object") {
        schema.type = "object";
    }
    if (!schema.properties || typeof schema.properties !== "object") {
        schema.properties = {};
    }
    if (!Array.isArray(schema.required)) {
        schema.required = [];
    }
    return schema;
}

class ToolFormRenderer {
    constructor(metadataContainer, argumentContainer) {
        this.metadataContainer = metadataContainer;
        this.argumentContainer = argumentContainer;
        this.metadataRenderer = null;
        this.argumentRenderer = null;
    }

    renderMetadata(seedData = {}) {
        this.metadataRenderer = new SchemaFormRenderer(this.metadataContainer, metadataSchema, seedData || {});
        this.metadataRenderer.render();
    }

    renderArgumentPreview(inputSchema, seedData = {}) {
        const normalized = coerceObjectSchema(inputSchema);
        this.argumentRenderer = new SchemaFormRenderer(this.argumentContainer, normalized, seedData || {});
        this.argumentRenderer.render();
    }

    metadataValues() {
        if (!this.metadataRenderer) {
            return null;
        }
        const validation = this.metadataRenderer.validate();
        if (!validation.valid) {
            return null;
        }
        return validation.value;
    }

    argumentPreviewValues() {
        if (!this.argumentRenderer) {
            return {};
        }
        const validation = this.argumentRenderer.validate();
        if (!validation.valid) {
            return {};
        }
        return validation.value;
    }
}

export { ToolFormRenderer, coerceObjectSchema };
