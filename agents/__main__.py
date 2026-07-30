from __future__ import annotations

import argparse
import asyncio
from typing import AsyncGenerator

from dotenv import load_dotenv

from agents.client import gateway
from agents.shared.storage import create_thread, init_db
from agents.shared.logging import (
    get_agent_logger,
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


async def invoke_conversation_stream(
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
        # node = update.get("node")
        # update_data = update.get("update", {})

        yield message if message else ""

        # if not isinstance(update_data, dict) and not message_content:
        #     continue

        # output_candidate = None

        # if message_content:
        #     if isinstance(message_content, str) and len(message_content.strip()) > 15:
        #         output_candidate = message_content.strip()
        # elif isinstance(update_data, dict):
        #     if "final_answer" in update_data and isinstance(update_data["final_answer"], str):
        #         output_candidate = update_data["final_answer"].strip()
        #         if len(output_candidate) > 15:
        #             yield output_candidate
        #             return
        #     elif "findings" in update_data and isinstance(update_data["findings"], str):
        #         output_candidate = f"Step {update_data.get('iteration', 0)}: {update_data['findings'].strip()}"
        #         if len(output_candidate.strip()) > 10:
        #             yield output_candidate
        #             continue
        #     elif "messages" in update_data:
        #         messages = update_data.get("messages", [])
        #         if messages:
        #             for msg in reversed(messages):
        #                 if hasattr(msg, 'role') and hasattr(msg, 'content'):
        #                     if msg.role == "assistant" and msg.content and isinstance(msg.content, str):
        #                         output_candidate = msg.content.strip()
        #                         if len(output_candidate) > 15:
        #                             yield output_candidate
        #                             break
        #                 elif isinstance(msg, dict) and msg.get("role") == "assistant":
        #                     output_candidate = msg.get("content", "").strip()
        #                     if len(output_candidate) > 15:
        #                         yield output_candidate
        #                         break
        #                 elif isinstance(msg, dict) and msg.get("role") == "human":
        #                     output_candidate = f"You: {msg.get('content', '').strip()}"
        #                     if len(output_candidate) > 0:
        #                         yield output_candidate
        #                         break

        # if output_candidate:
        #     yield output_candidate


async def cli_chat(
    agent_name: str = "supervisor",
    thread_id: str = "1",
    stream_mode: str = "invoke",
) -> None:
    """Start an interactive chat with an agent."""

    # Initialize the database and create a new thread for the CLI session if needed
    # await init_db()

    # save_graphs() is commented out to avoid saving graphs everytime, as it may not be necessary always.
    # gateway.save_graphs()

    mode_str = "stream" if stream_mode == "stream" else "invoke"
    logger.info("Starting CLI chat: agent=%s thread=%s mode=%s", agent_name, thread_id, mode_str)
    if thread_id == "1":
        thread_id = str(await create_thread("CLI session"))

    agents = await gateway.registered_agents()
    logger.info("Live Chat")
    logger.info("Agents: %s", ", ".join(agents.keys()))
    logger.info("Thread ID: %s", thread_id)
    logger.info("Type 'exit' to quit.")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.info("CLI chat interrupted by user")
            logger.info("Goodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            logger.info("CLI chat exit requested")
            logger.info("Goodbye.")
            break

        logger.debug("User input received: %s", user_input)
        if stream_mode == "stream":
            async for word in invoke_conversation_stream(user_input, thread_id, agent_name):
                print(f"{word}", end="", flush=True)
                logger.info("Streamed word (agent=%s thread=%s): %s", agent_name, thread_id, word)
            print("\n", end="")
            logger.debug("Finished streaming for input")
        else:
            response_accum = ""
            async for word in invoke_conversation(user_input, thread_id, agent_name):
                response_accum += word

            print(f'{agent_name} Agent: {response_accum}', end='\n\n')
            logger.info("Assistant response (agent=%s thread=%s): %s", agent_name, thread_id, response_accum)
            logger.debug("Finished streaming response for input")


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
        logger.info("Running single-message invocation: agent=%s message=%s", args.agent, args.message)
        if args.mode == "stream":
            async for word in invoke_conversation_stream(
                args.message,
                thread_id=args.thread_id,
                agent_name=args.agent,
            ):
                print(f"{args.agent} Agent: {word}", end="", flush=True)
                logger.info("Streamed word (agent=%s thread=%s): %s", args.agent, args.thread_id, word)
            print("\n", end="")
        else:
            response_accum = ""
            async for token in invoke_conversation(
                args.message,
                thread_id=args.thread_id,
                agent_name=args.agent,
            ):
                response_accum += token

            logger.info("Invocation result (agent=%s thread=%s): %s", args.agent, args.thread_id, response_accum)
            print(f"{args.agent} Agent: {response_accum}")


if __name__ == "__main__":
    asyncio.run(main())