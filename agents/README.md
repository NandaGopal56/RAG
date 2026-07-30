# Agents

This package contains the agent runtime used by the Kiro project.
It exposes one shared entry point for both direct agent calls and supervisor routing.

## Agents

- `personal`: general assistant agent for lightweight responses.
- `deep_research`: research agent for longer, more structured investigation.
- `supervisor`: routes a task to the appropriate agent.

## Main Entry Points

- `agents.client.gateway`: programmatic entry point for invoking any agent.
- `python -m agents`: CLI entry point for chat or one-shot messages.
- `python -m agents.test`: smoke test runner for the package.

## Commands

Run interactive chat:

```bash
uv run python -m agents --chat --agent supervisor
```

Run interactive chat with streaming:

```bash
uv run python -m agents --chat --agent supervisor --mode stream
```

Send a single message to an individual agent:

```bash
uv run python -m agents --agent personal --message "Say hello."
```

Send a message to an individual agent with streaming:

```bash
uv run python -m agents --agent personal --message "Say hello." --mode stream
```

Send a message to an individual agent with a specific thread:

```bash
uv run python -m agents --agent deep_research --thread-id 101 --message "Compare batteries and hydrogen for transport."
```

Send a message to the supervisor with a specific thread:

```bash
uv run python -m agents --agent supervisor --thread-id 201 --message "Route this task to the right agent."
```

Send a message to the supervisor with streaming:

```bash
uv run python -m agents --agent supervisor --thread-id 201 --message "Route this task to the right agent." --mode stream
```

Run smoke tests:

```bash
uv run python -m agents.test
```

## Notes

- Use a numeric-compatible `thread_id` when possible.
- The supervisor uses the same gateway as the direct agents.
- The `agents/0_legacy` folder is kept only for older compatibility code.
