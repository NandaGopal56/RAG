# Kiro

Kiro is a modular AI platform for combining speech, agents, retrieval, and orchestration into a flexible pipeline. The core idea is to keep each subsystem independently runnable while letting a thin runtime layer wire them together when needed.

## What Kiro is

Kiro brings together:
- speech input and transcription,
- agent-driven reasoning and routing,
- context retrieval for grounded responses,
- and a lightweight orchestration layer for composing these pieces.

It is designed for experimentation and extension, so new agents, providers, or runtime flows can be added without collapsing the whole system into a single tightly coupled app.

## Core modules

- Kiro runtime: the adapter layer that connects STT output to agent input and coordinates the interaction flow.
- Agents: modular agents including a supervisor, personal assistant, and deep research agent.
- STT: speech-to-text providers for turning audio into transcripts.
- TTS: text-to-speech support for spoken responses.
- Context Engine: document loading, chunking, embeddings, retrieval, and evaluation.
- Communication Bus: lightweight message passing between components.
- Shared utilities: common logging and supporting helpers.

## Key capabilities

- Independent modules that can run on their own.
- A normalized event flow for text and speech inputs.
- Pluggable providers for STT, TTS, embeddings, and stores.
- Agent-based orchestration with supervisor routing and specialized assistants.
- Retrieval-focused context building for RAG-style workflows.
- A simple path for future expansion into more modalities, agents, or backends.

## Getting started with uv

This project is designed to be run with uv, which handles environment creation and dependency installation cleanly.

### 1. Prerequisites

- Python 3.12+
- uv installed on your machine

### 2. Install dependencies

From the repository root, install the core project and the extras you need:

```bash
uv sync --extra dev --extra agents --extra stt --extra tts --extra context --extra ui
```

If you want a lighter setup for just the runtime flow, you can start with the core dependencies and only the extras you need.

### 3. Set up environment variables

Some modules rely on provider credentials such as OpenAI, Sarvam, or Tavily. Make sure the relevant environment variables are available before running the corresponding modules.

### 4. Run the Kiro runtime

You do not need to install the project as a package to try the runtime. The simplest path is to run the module directly:

```bash
uv run -m kiro
```

That starts the adapter flow and bridges STT output into the agent layer.

### 5. Run other modules independently

Because the system is modular, you can also run each subsystem on its own:

```bash
uv run -m stt
uv run -m agents
uv run -m context_engine
uv run -m communication_bus --smoke
```

These commands let you work with each subsystem independently without needing a single installable entry point.

### 6. Useful development commands

```bash
uv run pytest
uv run python -m pytest
uv run python -c "import kiro"
```

This makes it easy to develop, test, and run individual parts of the stack without forcing everything into a single entry point.

## System architecture

The full layered architecture (HAL, perception, world model, cognition, skill
registry, executor, orchestrator) and data-flow designs live in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Read it before adding hardware
(e.g. an arm) or new robot capabilities.

## Project structure

```text
agents/           # agent implementations and routing logic
stt/               # speech-to-text providers and engines
tts/               # text-to-speech support
context_engine/    # loaders, chunkers, embedders, stores, evals
communication_bus/ # messaging and event flow
kiro/              # runtime/adaptor layer for composing modules
shared/            # shared helpers and utilities
```

This structure is intentionally modular: each subsystem can evolve on its own while the runtime layer remains thin and composable.
