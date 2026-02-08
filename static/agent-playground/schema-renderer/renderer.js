import { validateAgainstSchema } from "./validators.js";

class SchemaFormRenderer {
    constructor(containerElement, schema, seedData = {}) {
        this.containerElement = containerElement;
        this.schema = (schema && typeof schema === "object") ? schema : { type: "object", properties: {} };
        this.seedData = (seedData && typeof seedData === "object") ? seedData : {};
        this.fields = new Map();
        this.errorNode = null;
    }

    _createField(name, fieldSchema) {
        const wrapper = document.createElement("div");
        wrapper.className = "ap-schema-field";

        const label = document.createElement("label");
        label.setAttribute("for", `schema-field-${name}`);
        label.textContent = name;
        wrapper.appendChild(label);

        const fieldType = String(fieldSchema?.type || "string");
        let input;

        if (Array.isArray(fieldSchema?.enum)) {
            input = document.createElement("select");
            for (const optionValue of fieldSchema.enum) {
                const option = document.createElement("option");
                option.value = String(optionValue);
                option.textContent = String(optionValue);
                input.appendChild(option);
            }
        } else if (fieldType === "boolean") {
            input = document.createElement("input");
            input.type = "checkbox";
        } else if (fieldType === "integer" || fieldType === "number") {
            input = document.createElement("input");
            input.type = "number";
            if (typeof fieldSchema.minimum === "number") {
                input.min = String(fieldSchema.minimum);
            }
            if (typeof fieldSchema.maximum === "number") {
                input.max = String(fieldSchema.maximum);
            }
            if (fieldType === "integer") {
                input.step = "1";
            }
        } else if (fieldType === "array") {
            input = document.createElement("textarea");
            input.rows = 3;
            input.placeholder = "One value per line";
        } else if (fieldType === "object") {
            input = document.createElement("textarea");
            input.rows = 5;
            input.placeholder = "JSON object";
        } else {
            input = document.createElement("input");
            input.type = "text";
        }

        input.id = `schema-field-${name}`;
        input.name = name;
        if (typeof fieldSchema?.description === "string" && fieldSchema.description.trim()) {
            input.title = fieldSchema.description;
        }

        const seedValue = this.seedData[name] ?? fieldSchema?.default;
        if (seedValue !== undefined) {
            if (input.type === "checkbox") {
                input.checked = Boolean(seedValue);
            } else if (Array.isArray(seedValue)) {
                input.value = seedValue.join("\n");
            } else if (fieldType === "object" && seedValue && typeof seedValue === "object") {
                input.value = JSON.stringify(seedValue, null, 2);
            } else {
                input.value = String(seedValue);
            }
        }

        wrapper.appendChild(input);
        this.fields.set(name, { input, schema: fieldSchema || {} });
        return wrapper;
    }

    render() {
        if (!this.containerElement) {
            return;
        }

        this.containerElement.innerHTML = "";
        this.containerElement.id = this.containerElement.id || "ap-schema-form";

        const properties = (this.schema.properties && typeof this.schema.properties === "object")
            ? this.schema.properties
            : {};

        for (const [name, fieldSchema] of Object.entries(properties)) {
            this.containerElement.appendChild(this._createField(name, fieldSchema));
        }

        const errorNode = document.createElement("div");
        errorNode.className = "ap-validation-error";
        errorNode.style.display = "none";
        this.containerElement.appendChild(errorNode);
        this.errorNode = errorNode;
    }

    getValue() {
        const result = {};
        for (const [name, field] of this.fields.entries()) {
            const input = field.input;
            const fieldType = String(field.schema?.type || "string");

            if (input.type === "checkbox") {
                result[name] = Boolean(input.checked);
                continue;
            }

            const raw = String(input.value || "").trim();
            if (raw === "") {
                continue;
            }

            if (fieldType === "integer") {
                const parsed = Number.parseInt(raw, 10);
                if (!Number.isNaN(parsed)) {
                    result[name] = parsed;
                }
                continue;
            }

            if (fieldType === "number") {
                const parsed = Number.parseFloat(raw);
                if (!Number.isNaN(parsed)) {
                    result[name] = parsed;
                }
                continue;
            }

            if (fieldType === "array") {
                const lines = raw
                    .split(/\r?\n/)
                    .map((item) => item.trim())
                    .filter((item) => item.length > 0);
                result[name] = lines;
                continue;
            }

            if (fieldType === "object") {
                try {
                    result[name] = JSON.parse(raw);
                } catch (_error) {
                    result[name] = {};
                }
                continue;
            }

            result[name] = raw;
        }

        return result;
    }

    validate() {
        const value = this.getValue();
        const errors = validateAgainstSchema(value, this.schema);
        if (this.errorNode) {
            if (errors.length === 0) {
                this.errorNode.style.display = "none";
                this.errorNode.textContent = "";
            } else {
                this.errorNode.style.display = "block";
                this.errorNode.textContent = errors.join(" ");
            }
        }
        return {
            valid: errors.length === 0,
            errors,
            value,
        };
    }
}

export { SchemaFormRenderer };
