function diffObjects(beforeValue, afterValue) {
    const before = (beforeValue && typeof beforeValue === "object") ? beforeValue : {};
    const after = (afterValue && typeof afterValue === "object") ? afterValue : {};

    const changes = {
        added: {},
        removed: {},
        changed: {},
    };

    for (const key of Object.keys(after)) {
        if (!(key in before)) {
            changes.added[key] = after[key];
            continue;
        }
        const beforeItem = before[key];
        const afterItem = after[key];
        if (JSON.stringify(beforeItem) !== JSON.stringify(afterItem)) {
            changes.changed[key] = {
                before: beforeItem,
                after: afterItem,
            };
        }
    }

    for (const key of Object.keys(before)) {
        if (!(key in after)) {
            changes.removed[key] = before[key];
        }
    }

    return changes;
}

export { diffObjects };
