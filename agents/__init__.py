"""Agents package."""

__all__ = ["invoke_conversation"]


def __getattr__(name: str):
    from agents.shared.logging import get_agent_logger

    _logger = get_agent_logger("agents", "init")
    _logger.info("LAZY_IMPORT name=%s", name)

    if name == "invoke_conversation":
        from .__main__ import invoke_conversation

        return invoke_conversation

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
