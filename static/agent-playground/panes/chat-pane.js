class ChatPane {
    constructor(containerElement) {
        this.container = containerElement;
        this.activeAssistantMessage = null;
        this._renderEmpty();
    }

    _renderEmpty() {
        if (!this.container) {
            return;
        }
        if (this.container.children.length === 0) {
            const node = document.createElement("div");
            node.className = "ap-empty";
            node.textContent = "No chat activity yet. Start a run to stream output.";
            this.container.appendChild(node);
        }
    }

    _appendMessage(role, text) {
        if (!this.container) {
            return null;
        }

        const emptyNode = this.container.querySelector(".ap-empty");
        if (emptyNode) {
            emptyNode.remove();
        }

        const message = document.createElement("div");
        message.className = `ap-chat-message ${role}`;
        message.textContent = text;
        this.container.appendChild(message);
        this.container.scrollTop = this.container.scrollHeight;
        return message;
    }

    addUserMessage(text) {
        this.activeAssistantMessage = null;
        return this._appendMessage("user", String(text || ""));
    }

    addSystemMessage(text) {
        this.activeAssistantMessage = null;
        return this._appendMessage("system", String(text || ""));
    }

    beginAssistantMessage() {
        this.activeAssistantMessage = this._appendMessage("assistant", "");
        return this.activeAssistantMessage;
    }

    appendAssistantChunk(text) {
        const chunk = String(text || "");
        if (!chunk) {
            return;
        }
        if (!this.activeAssistantMessage) {
            this.beginAssistantMessage();
        }
        this.activeAssistantMessage.textContent += chunk;
        this.container.scrollTop = this.container.scrollHeight;
    }

    finalizeAssistantMessage(fullText) {
        const text = String(fullText || "");
        if (!text) {
            return;
        }
        if (!this.activeAssistantMessage) {
            this.beginAssistantMessage();
        }
        this.activeAssistantMessage.textContent = text;
        this.container.scrollTop = this.container.scrollHeight;
    }

    reset() {
        if (!this.container) {
            return;
        }
        this.container.innerHTML = "";
        this.activeAssistantMessage = null;
        this._renderEmpty();
    }
}

export { ChatPane };
