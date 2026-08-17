"""Agents package."""

__all__ = ["AgentProcessor", "create_service", "invoke_conversation"]


def __getattr__(name: str):
    from agents.shared.logging import get_agent_logger, log_event

    _logger = get_agent_logger("agents", "init")
    log_event(_logger, "LAZY_IMPORT", name=name)

    if name in {"AgentProcessor", "create_service"}:
        from .legacy.agent_processor import AgentProcessor, create_service

        return {"AgentProcessor": AgentProcessor, "create_service": create_service}[name]

    if name == "invoke_conversation":
        from .bot import invoke_conversation

        return invoke_conversation

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
