from __future__ import annotations

import argparse
import asyncio
from typing import AsyncGenerator

from dotenv import load_dotenv

from agents.client import gateway
from agents.shared.storage import create_thread, init_db
from agents.shared.logging import get_agent_logger

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
    logger.info("INVOKE_START agent=%s thread=%s mode=cli task=%s", agent_name, thread_id, (message[:200] + "...") if len(message) > 200 else message)
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
    logger.info("INVOKE_START agent=%s thread=%s mode=cli task=%s", agent_name, thread_id, (message[:200] + "...") if len(message) > 200 else message)

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
    logger.info("CLI_START agent=%s thread=%s mode=%s", agent_name, thread_id, mode_str)
    if thread_id == "1":
        thread_id = str(await create_thread("CLI session"))

    agents = await gateway.registered_agents()
    logger.info("CLI_AGENTS agents=%s thread=%s", list(agents.keys()), thread_id)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.info("CLI_INTERRUPT agent=%s thread=%s", agent_name, thread_id)
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            logger.info("CLI_EXIT agent=%s thread=%s", agent_name, thread_id)
            break

        logger.info("CLI_USER_INPUT agent=%s thread=%s preview=%s", agent_name, thread_id, user_input[:200])
        if stream_mode == "stream":
            async for word in stream_conversation(user_input, thread_id, agent_name):
                print(f"{word}", end="", flush=True)
                logger.log(10, "CLI_STREAM_WORD agent=%s thread=%s word=%s", agent_name, thread_id, word)
            print("\n", end="")
            logger.info("CLI_STREAM_DONE agent=%s thread=%s", agent_name, thread_id)
        else:
            response_accum = ""
            async for word in invoke_conversation(user_input, thread_id, agent_name):
                response_accum += word

            print(f'{agent_name} Agent: {response_accum}', end='\n\n')
            logger.info("CLI_RESPONSE agent=%s thread=%s response_length=%s", agent_name, thread_id, len(response_accum))


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
        logger.info("CLI_SINGLE_START agent=%s message=%s thread=%s", args.agent, args.message[:200], args.thread_id)
        if args.mode == "stream":
            async for word in stream_conversation(
                args.message,
                thread_id=args.thread_id,
                agent_name=args.agent,
            ):
                print(f"{args.agent} Agent: {word}", end="", flush=True)
                logger.log(10, "CLI_STREAM_WORD agent=%s thread=%s word=%s", args.agent, args.thread_id, word)
            print("\n", end="")
        else:
            response_accum = ""
            async for token in invoke_conversation(
                args.message,
                thread_id=args.thread_id,
                agent_name=args.agent,
            ):
                response_accum += token

            logger.info("CLI_SINGLE_RESULT agent=%s thread=%s response_length=%s", args.agent, args.thread_id, len(response_accum))
            print(f"{args.agent} Agent: {response_accum}")


if __name__ == "__main__":
    asyncio.run(main())
