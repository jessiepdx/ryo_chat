# RYO Chat  
Run Your Own LLM-Powered Chat Interfaces  

[![GitHub license](https://img.shields.io/github/license/jessiepdx/ryo_chat)](LICENSE)  
[![GitHub issues](https://img.shields.io/github/issues/jessiepdx/ryo_chat)](https://github.com/jessiepdx/ryo_chat “GitHub Issues”)

RYO Chat is an open-source framework for building flexible, multi-interface chat systems powered by large language models (LLMs). Built with a modular architecture, RYO Chat enables 
users to interact with configurable agents across various platforms, including Telegram, CLI, WebApp, and Twitter. The project emphasizes open-source compliance, leveraging tools like 
Ollama for model inference and PostgreSQL for data management.  

RYO Chat empowers users to customize their chat experience by selecting preferred LLMs, defining agent behaviors, and integrating specialized tools. Whether you’re managing group 
conversations or conducting private interactions, RYO Chat provides a scalable and extensible platform for conversational AI.  

---

## Features  

### **Core Architecture**  
- **Modular Design**: Built with interchangeable components (agents, tools, interfaces).  
- **Agent-Based Conversations**: Leverages specialized agents for message analysis, tool integration, and conversational flow.  
- **Flexible Interfaces**: Supports Telegram (full-featured), CLI/TUI (lightweight), WebApp (browser-based), and Twitter (social media integration).  

### **Key Technologies**  
- **Open-Source LLMs**: Uses Ollama to run open-weight models (e.g., LLaMA, Mistral, DeepSeek) for transparent and customizable inference.  
- **PostgreSQL**: Handles relational data (e.g., user profiles) and vector embeddings for retrieval-augmented generation (RAG).  
- **Retrieval-Augmented Generation (RAG)**: Integrates domain-specific knowledge retrieval for contextual responses.  
- **Brave Web Search**: Fetches real-time, external information for up-to-date answers.  

### **Use Cases**  
- **Multi-User Conversations**: Agents facilitate group chats while maintaining context.  
- **Customizable Agents**: Administrators can configure agent behavior via JSON policy files.  
- **Flexible Deployment**: Deploy RYO Chat across platforms, from local machines to cloud environments. 

---

## Agents & Tools  

### **Built-in Agents**  
RYO Chat includes pre-configured agents for:  
- **Message Analysis**: Parses user input to extract intent.  
- **Tool Integration**: Coordinates the use of external tools (e.g., search, RAG).  
- **Conversational Response**: Generates natural, context-aware replies.  

### **Custom Agents**  
While currently maintained by developers, administrators can extend RYO Chat by:  
1. Defining new agents in JSON format.  
2. Specifying model preferences, system prompts, and tool dependencies.  

### **Built-in Tools**  
- **RAG Knowledge Retrieval**: Searches a domain-specific knowledge base for contextual replies.  
- **Brave Web Search**: Queries external sources for real-time information.  
- **Chat History Lookup**: Retrieves past conversations for continuity.

## Upcoming updates

## Requirements