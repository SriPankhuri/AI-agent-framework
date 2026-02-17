# AI Agent Framework (Local Offline LLM Agent)

## Overview

This project is a **modular AI agent framework** that runs a local Large Language Model (LLM) on your own computer (offline after first download).

Instead of behaving like a simple chatbot, the system acts like a **mini AI operating system**:

User Input → Planner → Tasks → Tools → Memory → Final Report

It demonstrates how real autonomous AI agents are built in production systems.

---

## Key Features

* Fully local AI (no OpenAI / API required after setup)
* Workflow‑based agent execution (TaskFlow DAG)
* Planner that decomposes goals into steps
* Tool execution system (ToolRegistry)
* Persistent memory (SQLite audit trail)
* Controller orchestrates reasoning loop
* Replaceable LLM backend (MockLLM or real HuggingFace model)
* Modular architecture similar to LangChain / AutoGPT style agents

---

## Tech Stack

| Component    | Technology                                     |
| ------------ | ---------------------------------------------- |
| Language     | Python 3.10+                                   |
| LLM          | Microsoft Phi‑3 Mini (local HuggingFace model) |
| Persistence  | SQLite                                         |
| Architecture | Agentic workflow framework                     |
| ML Backend   | Transformers + PyTorch                         |
| Tokenizer    | HuggingFace Tokenizers                         |

---

## Folder Structure

```
AI-agent-framework/
│
├── main.py                 # Entry point
├── agent/
│   ├── controller.py       # Main orchestrator (brain)
│   ├── planner.py          # Breaks goal into tasks
│   ├── flow.py             # TaskFlow + Task definitions
│   └── memory.py           # Persistent memory system
│
├── llm/
│   ├── llm_client.py       # Real local model loader
│   ├── mock_llm.py         # Fake model for testing
│   └── llm_client_base.py  # LLM interface abstraction
│
├── tools/
│   └── tool_registry.py    # Tool execution engine
│
├── observability/
│   └── logger.py           # Logging & monitoring
│
└── agent_audit_trail.db    # Auto‑generated memory database
```

---

## How The Agent Works

1. User provides a goal
2. Planner converts goal → list of tasks
3. Controller executes tasks sequentially
4. Tools perform actions
5. Memory records results
6. LLM generates final report

---

## First Time Setup (Required Only Once)

### 1. Install Python

Install Python 3.10 or newer
[https://www.python.org/downloads/](https://www.python.org/downloads/)

During installation enable:
"Add Python to PATH"

---

### 2. Install Required Libraries

Open terminal inside project folder:

```
pip install torch transformers accelerate huggingface_hub sentencepiece
```

(Optional but faster download)

```
pip install huggingface_hub[hf_xet]
```

---

### 3. Login to HuggingFace (Required for model download)

```
hf auth login
```

Paste token from:
[https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

Give READ permission only.

---

### 4. First Run (Downloads AI Model ~7GB)

```
python -m main
```

This will download:
Microsoft Phi‑3 Mini 4K Instruct model

Time depends on internet speed (can take hours).

IMPORTANT: Do not close terminal during first download.

After download completes → future runs are instant.

---

## Running The Project (After Setup)

Just run:

```
python -m main
```

No internet needed anymore.

---

## Expected Output

```
--- Starting Agentic Workflow ---
Routing task 'market_research'
Routing task 'summary'

===== FINAL AGENT REPORT =====
User Request: ...
Workflow Results: ...
```

---

## Running On Another Computer

To run on another PC you have two options:

### Option A — Fresh Setup (Recommended)

1. Install Python
2. Clone repo
3. Install libraries
4. Login HuggingFace
5. Run once (downloads model)

### Option B — Copy Model (Fast)

Copy this folder from original PC:

Windows Path:

```
C:\Users\<username>\.cache\huggingface\hub\models--microsoft--Phi-3-mini-4k-instruct
```

Paste into same location on new PC.

Then run:

```
python -m main
```

(No download required)

---

## Disk & RAM Requirements

| Resource | Required                         |
| -------- | -------------------------------- |
| Storage  | 10 GB free                       |
| RAM      | 8 GB minimum (16 GB recommended) |
| GPU      | Not required                     |

---

## Why This Project Matters

This project demonstrates real AI engineering concepts:

* Agent architecture
* Planning & execution loops
* Tool calling
* Memory persistence
* Offline LLM integration

It is closer to building your own ChatGPT than using an API.

---

## Future Improvements

* Add real web search tool
* Add multi‑agent collaboration
* Add voice interface
* Add web UI

---

## Author

Educational AI agent framework project for learning how autonomous AI systems work locally without cloud APIs.
## Contribution
Fixed runtime errors and made framework executable locally.
