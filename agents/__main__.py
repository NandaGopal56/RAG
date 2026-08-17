from __future__ import annotations

import argparse
import asyncio
from typing import AsyncGenerator

from dotenv import load_dotenv

from agents.client import gateway
from agents.shared.storage import create_thread, init_db
from agents.shared.logging import (
    get_agent_logger,
    log_event,
    log_invoke_start,
)

load_dotenv()

logger = get_agent_logger("cli", "__main__")


def _split_response_words(response_text: str) -> AsyncGenerator[str, None]:
    async def _generator():
        words = response_text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0)

    return _generator()


async def invoke_conversation(
    message: str,
    thread_id: str = "1",
    agent_name: str = "supervisor",
) -> AsyncGenerator[str, None]:
    log_invoke_start(logger, agent_name, thread_id=thread_id, mode="cli", task_preview=message)
    response_text = await gateway.invoke(
        agent_name=agent_name,
        task=message,
        thread_id=thread_id,
    )
    if response_text:
        async for word in _split_response_words(response_text):
            yield word


async def stream_conversation(
    message: str,
    thread_id: str = "1",
    agent_name: str = "supervisor",
) -> AsyncGenerator[str, None]:
    log_invoke_start(logger, agent_name, thread_id=thread_id, mode="cli", task_preview=message)

    async for message in gateway.stream(
        agent_name=agent_name,
        task=message,
        thread_id=thread_id,
    ):
        yield message if message else ""


async def cli_chat(
    agent_name: str = "supervisor",
    thread_id: str = "1",
    stream_mode: str = "invoke",
) -> None:
    """Start an interactive chat with an agent."""

    mode_str = "stream" if stream_mode == "stream" else "invoke"
    log_event(logger, "CLI_START", agent=agent_name, thread=thread_id, mode=mode_str)
    if thread_id == "1":
        thread_id = str(await create_thread("CLI session"))

    agents = await gateway.registered_agents()
    log_event(logger, "CLI_AGENTS", agents=list(agents.keys()), thread=thread_id)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            log_event(logger, "CLI_INTERRUPT", agent=agent_name, thread=thread_id)
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            log_event(logger, "CLI_EXIT", agent=agent_name, thread=thread_id)
            break

        log_event(logger, "CLI_USER_INPUT", agent=agent_name, thread=thread_id, preview=user_input[:200])
        if stream_mode == "stream":
            async for word in stream_conversation(user_input, thread_id, agent_name):
                print(f"{word}", end="", flush=True)
                log_event(logger, "CLI_STREAM_WORD", level=10, agent=agent_name, thread=thread_id, word=word)
            print("\n", end="")
            log_event(logger, "CLI_STREAM_DONE", agent=agent_name, thread=thread_id)
        else:
            response_accum = ""
            async for word in invoke_conversation(user_input, thread_id, agent_name):
                response_accum += word

            print(f'{agent_name} Agent: {response_accum}', end='\n\n')
            log_event(logger, "CLI_RESPONSE", agent=agent_name, thread=thread_id, response_length=len(response_accum))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agents package.")
    parser.add_argument(
        "--agent",
        choices=["supervisor", "personal", "deep_research"],
        default="supervisor",
        help="Which agent to invoke.",
    )
    parser.add_argument(
        "--message",
        default="hi",
        help="Send one message and print the response.",
    )
    parser.add_argument("--thread-id", default="1")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat.")
    parser.add_argument("--stream", action="store_true", help="Stream responses incrementally in chat mode.")
    parser.add_argument(
        "--mode",
        choices=["invoke", "stream"],
        default="invoke",
        help="Mode: 'invoke' for full response or 'stream' for incremental updates.",
    )
    args = parser.parse_args()

    if args.chat:
        await cli_chat(agent_name=args.agent, thread_id=args.thread_id, stream_mode=args.mode)
    else:
        log_event(logger, "CLI_SINGLE_START", agent=args.agent, message=args.message[:200], thread=args.thread_id)
        if args.mode == "stream":
            async for word in stream_conversation(
                args.message,
                thread_id=args.thread_id,
                agent_name=args.agent,
            ):
                print(f"{args.agent} Agent: {word}", end="", flush=True)
                log_event(logger, "CLI_STREAM_WORD", level=10, agent=args.agent, thread=args.thread_id, word=word)
            print("\n", end="")
        else:
            response_accum = ""
            async for token in invoke_conversation(
                args.message,
                thread_id=args.thread_id,
                agent_name=args.agent,
            ):
                response_accum += token

            log_event(logger, "CLI_SINGLE_RESULT", agent=args.agent, thread=args.thread_id, response_length=len(response_accum))
            print(f"{args.agent} Agent: {response_accum}")


if __name__ == "__main__":
    asyncio.run(main())
