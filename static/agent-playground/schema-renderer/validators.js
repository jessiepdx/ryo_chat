function _typeOf(value) {
    if (value === null) {
        return "null";
    }
    if (Array.isArray(value)) {
        return "array";
    }
    return typeof value;
}

function _coerceUnionTypes(typeValue) {
    if (Array.isArray(typeValue)) {
        return typeValue.map((item) => String(item));
    }
    return [String(typeValue || "")];
}

function _matchesType(value, expected) {
    if (!expected || expected.length === 0) {
        return true;
    }

    const actual = _typeOf(value);
    if (expected.includes(actual)) {
        return true;
    }

    if (actual === "number" && expected.includes("integer")) {
        return Number.isInteger(value);
    }

    return false;
}

function validateAgainstSchema(value, schema) {
    const errors = [];
    const safeSchema = (schema && typeof schema === "object") ? schema : {};

    if (safeSchema.type) {
        const allowed = _coerceUnionTypes(safeSchema.type);
        if (!_matchesType(value, allowed)) {
            errors.push(`Expected type ${allowed.join("|")} but received ${_typeOf(value)}.`);
        }
    }

    if (safeSchema.type === "object" || (safeSchema.properties && typeof value === "object" && value !== null && !Array.isArray(value))) {
        const required = Array.isArray(safeSchema.required) ? safeSchema.required : [];
        for (const key of required) {
            const fieldValue = value?.[key];
            if (fieldValue === undefined || fieldValue === null || (typeof fieldValue === "string" && fieldValue.trim() === "")) {
                errors.push(`Missing required field: ${key}`);
            }
        }

        const properties = (safeSchema.properties && typeof safeSchema.properties === "object") ? safeSchema.properties : {};
        for (const [field, fieldSchema] of Object.entries(properties)) {
            if (value?.[field] === undefined || value?.[field] === null) {
                continue;
            }
            const childErrors = validateAgainstSchema(value[field], fieldSchema);
            for (const childError of childErrors) {
                errors.push(`${field}: ${childError}`);
            }
        }
    }

    if (safeSchema.type === "array" && Array.isArray(value)) {
        if (typeof safeSchema.minItems === "number" && value.length < safeSchema.minItems) {
            errors.push(`Expected at least ${safeSchema.minItems} items.`);
        }
        if (typeof safeSchema.maxItems === "number" && value.length > safeSchema.maxItems) {
            errors.push(`Expected at most ${safeSchema.maxItems} items.`);
        }

        if (safeSchema.items && typeof safeSchema.items === "object") {
            value.forEach((item, index) => {
                const itemErrors = validateAgainstSchema(item, safeSchema.items);
                for (const itemError of itemErrors) {
                    errors.push(`Item ${index + 1}: ${itemError}`);
                }
            });
        }
    }

    if (typeof value === "string") {
        if (typeof safeSchema.minLength === "number" && value.length < safeSchema.minLength) {
            errors.push(`Minimum length is ${safeSchema.minLength}.`);
        }
        if (typeof safeSchema.maxLength === "number" && value.length > safeSchema.maxLength) {
            errors.push(`Maximum length is ${safeSchema.maxLength}.`);
        }
        if (Array.isArray(safeSchema.enum) && safeSchema.enum.length > 0 && !safeSchema.enum.includes(value)) {
            errors.push(`Expected one of: ${safeSchema.enum.join(", ")}.`);
        }
    }

    if (typeof value === "number") {
        if (typeof safeSchema.minimum === "number" && value < safeSchema.minimum) {
            errors.push(`Minimum value is ${safeSchema.minimum}.`);
        }
        if (typeof safeSchema.maximum === "number" && value > safeSchema.maximum) {
            errors.push(`Maximum value is ${safeSchema.maximum}.`);
        }
    }

    return errors;
}

export { validateAgainstSchema };
