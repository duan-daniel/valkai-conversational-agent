import argparse
import uuid

from dotenv import load_dotenv

from agent.core import make_agent


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="CLI Chat Agent")
    parser.add_argument(
        "--model",
        default="anthropic:claude-haiku-4-5-20251001",
        help="Model string, e.g. openai:gpt-4o, anthropic:claude-haiku-4-5-20251001, google_genai:gemini-2.5-flash",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Custom system prompt",
    )
    parser.add_argument(
        "--memory",
        default="none",
        choices=["none", "session", "long_term"],
        help="Memory type: none (default), session (thread-scoped), long_term (cross-thread)",
    )
    args = parser.parse_args()

    agent = make_agent(args.model, args.system, memory_type=args.memory)
    use_checkpointer = args.memory in ("session", "long_term")

    if use_checkpointer:
        thread_id = uuid.uuid4().hex[:8] # generate a random UUID hex
        config = {"configurable": {"thread_id": thread_id}}
        print(f"Chat started (memory={args.memory}, thread={thread_id}). Type 'quit' to exit.\n")
    else:
        config = None
        messages = []
        print("Chat started. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        if use_checkpointer:
            # Checkpointer manages history; send only the new message
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
        else:
            # No checkpointer; caller accumulates full history
            messages.append({"role": "user", "content": user_input})
            result = agent.invoke({"messages": messages})
            messages = result["messages"]

        ai_msg = result["messages"][-1]
        print(f"\nAssistant: {ai_msg.content}\n")


if __name__ == "__main__":
    main()
