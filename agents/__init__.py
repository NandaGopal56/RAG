"""Agents package."""

__all__ = ["invoke_conversation"]


def __getattr__(name: str):
    from agents.shared.logging import get_agent_logger, log_event

    _logger = get_agent_logger("agents", "init")
    log_event(_logger, "LAZY_IMPORT", name=name)

    if name == "invoke_conversation":
        from .__main__ import invoke_conversation

        return invoke_conversation

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
