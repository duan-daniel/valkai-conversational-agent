"""Scripted conversation harness comparing memory types.

Runs three conversations (A, B, C) against each memory strategy and prints
results side-by-side so behavioral differences are immediately visible.

Usage:
    uv run harness
"""

from dotenv import load_dotenv

from agent.core import make_agent

# ── Conversation scripts ─────────────────────────────────────────────────────

CONV_A = [
    "Hi! My name is Daniel and I'm a software engineer.",
    "I love hiking and my favorite programming language is Python.",
    "I'm allergic to shellfish, please remember that.",
]

CONV_B = [
    "Hey, do you remember anything about me?",
    "What food should I avoid?",
    "What programming language do I prefer?",
]

CONV_C = [
    "What have we been talking about?",
    "What's my name again?",
]

MEMORY_TYPES = ["none", "session", "long_term"]

SEPARATOR = "─" * 72


def _run_conversation(agent, turns, *, config=None, use_checkpointer=False, messages=None):
    """Run a list of user turns and return (responses, updated_messages).

    For checkpointed agents, each turn sends only the new message.
    For baseline agents, the caller-managed message list is accumulated.

    `messages` is only returned to print out how many messages the baseline agent accumulated.
    `responses` is returned to print out each turn's AI response via `_print_turn`.
    """
    if messages is None:
        messages = []
    responses = []

    for turn in turns:
        if use_checkpointer:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": turn}]},
                config=config,
            )
        else:
            messages.append({"role": "user", "content": turn})
            result = agent.invoke({"messages": messages})
            messages = result["messages"]

        ai_msg = result["messages"][-1]
        responses.append(ai_msg.content)

    return responses, messages


def _print_section(title):
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}")


def _print_turn(user_msg, responses_by_type):
    """Print one user turn and each strategy's response side-by-side.

    responses_by_type: mem_type -> single AI response string for this turn
    """
    print(f"\n  User: \"{user_msg}\"")
    print(f"  {SEPARATOR}")
    for mem_type, response in responses_by_type.items():
        # Truncate very long responses for readability
        text = response[:300] + "..." if len(response) > 300 else response
        print(f"  [{mem_type:>10}]  {text}") # right-align mem_type in 10-character-wide field
    print()


def main():
    load_dotenv()

    print("\n" + "=" * 72)
    print("  MEMORY TYPE COMPARISON HARNESS")
    print("=" * 72)
    print("\nThis harness runs three conversations against each memory type")
    print("to demonstrate behavioral differences.\n")
    print("Memory types: baseline (none), session (checkpointer), long_term (store + tools)")
    print(f"{SEPARATOR}\n")

    # Build agents
        # mem_type -> compiled LangGraph agent (one per strategy: "none", "session", "long_term")
    agents = {}
    for mem_type in MEMORY_TYPES:
        print(f"  Creating agent with memory_type={mem_type!r}...")
        agents[mem_type] = make_agent(memory_type=mem_type)
    print()

    # ── Run Conversation A ────────────────────────────────────────────────

    _print_section("CONVERSATION A — Sharing Information (thread: conv-a)")
    print("  The user shares personal details. We observe how each agent responds.")

    # mem_type -> list of AI response strings, one per turn in CONV_A
    conv_a_responses = {}

    # mem_type -> accumulated message list from _run_conversation;
    # only meaningful for "none" (baseline), used later to report how many
    # messages the caller had to manually track
    conv_a_messages = {}

    # Run all 3 turns of CONV_A through each agent and collects the responses
    for mem_type in MEMORY_TYPES:
        use_cp = mem_type in ("session", "long_term")
        config = {"configurable": {"thread_id": "conv-a"}} if use_cp else None

        responses, messages = _run_conversation(
            agents[mem_type],
            CONV_A,
            config=config,
            use_checkpointer=use_cp,
        )
        conv_a_responses[mem_type] = responses
        conv_a_messages[mem_type] = messages

    # For each turn `i`, builds a dict of `memory_type` to `response` 
    for i, turn in enumerate(CONV_A):
        _print_turn(turn, {mt: conv_a_responses[mt][i] for mt in MEMORY_TYPES})

    # ── Run Conversation B ────────────────────────────────────────────────

    _print_section("CONVERSATION B — New Thread (thread: conv-b) — Testing Cross-Thread Recall")
    print("  A brand-new thread. Only long-term memory should recall user facts.")

    # mem_type -> list of AI response strings, one per turn for CONV_B
    conv_b_responses = {}

    for mem_type in MEMORY_TYPES:
        use_cp = mem_type in ("session", "long_term")
        config = {"configurable": {"thread_id": "conv-b"}} if use_cp else None

        responses, _ = _run_conversation(
            agents[mem_type],
            CONV_B,
            config=config,
            use_checkpointer=use_cp,
        )
        conv_b_responses[mem_type] = responses

    for i, turn in enumerate(CONV_B):
        _print_turn(turn, {mt: conv_b_responses[mt][i] for mt in MEMORY_TYPES})

    # ── Run Conversation C ────────────────────────────────────────────────

    _print_section("CONVERSATION C — Resume Thread A (thread: conv-a) — Testing Session Resumption")
    print("  Resuming the original thread. Session + long-term should recall; baseline should not.")

    # mem_type -> list of AI response strings, one per turn for CONV_C
    conv_c_responses = {}

    for mem_type in MEMORY_TYPES:
        use_cp = mem_type in ("session", "long_term")
        config = {"configurable": {"thread_id": "conv-a"}} if use_cp else None

        responses, _ = _run_conversation(
            agents[mem_type],
            CONV_C,
            config=config,
            use_checkpointer=use_cp,
        )
        conv_c_responses[mem_type] = responses

    for i, turn in enumerate(CONV_C):
        _print_turn(turn, {mt: conv_c_responses[mt][i] for mt in MEMORY_TYPES})

    # ── Storage inspection ─────────────────────────────────────────────────

    _print_section("WHAT EACH MEMORY TYPE STORED")
    print("  Comparing what is stored under the hood after Conversation A.\n")

    # Baseline: nothing stored
    print(f"  [      none]  Nothing — no persistence mechanism. The caller held")
    print(f"                {len(conv_a_messages['none'])} messages in a Python list during the session.\n")

    # Session: full checkpoint state
    session_agent = agents["session"]
    session_config = {"configurable": {"thread_id": "conv-a"}}

    # state is a StateSnapshot object
    state = session_agent.get_state(session_config)

    # state.values is a dict of all the graph's channel values at the latest checkpoint. 
        # channels are named slots that hold the graph's state — graph reads from + writes to these channels as nodes execute.
    # since this is a MessagesState graph, "messages" key holds the full conversation history
        # create_deep_agent uses MessagesState internally
    # that the checkpointer has been automatically accumulating.
    checkpoint_msgs = state.values.get("messages", [])
    print(f"  [   session]  Checkpointer stored the full graph state for thread 'conv-a'.")
    print(f"                Contains {len(checkpoint_msgs)} messages (every turn, verbatim):")
    for msg in checkpoint_msgs:
        label = msg.type.upper()
        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"                  [{label:>9}] {content}")
    print()

    # Long-term: selective key-value pairs in the store
    lt_agent = agents["long_term"]
    lt_store = lt_agent.store
    stored_items = lt_store.search(("user_memories",), limit=20)
    print(f"  [ long_term]  Store contains {len(stored_items)} selectively extracted facts:")
    for item in stored_items:
        print(f"                  {item.key}: {item.value['value']}")
    print()
    print(f"                (The agent also has a checkpointer with the full message history,")
    print(f"                 but the store is what enables cross-thread recall.)")
    print()

    # ── Summary ───────────────────────────────────────────────────────────

    _print_section("SUMMARY")
    print()
    print("  Expected behavior:")
    print()
    print("  ┌──────────────┬─────────────────────────┬──────────────────────────────┐")
    print("  │ Memory Type  │ Conv B (new thread)     │ Conv C (resume thread A)     │")
    print("  ├──────────────┼─────────────────────────┼──────────────────────────────┤")
    print("  │ none         │ No recall               │ No recall                    │")
    print("  │ session      │ No recall               │ Full recall                  │")
    print("  │ long_term    │ Recalls stored facts    │ Full recall                  │")
    print("  └──────────────┴─────────────────────────┴──────────────────────────────┘")
    print()


if __name__ == "__main__":
    main()
