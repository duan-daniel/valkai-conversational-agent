from __future__ import annotations

from typing import Annotated, Any

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import InjectedStore
from langgraph.store.memory import InMemoryStore

from deepagents import create_deep_agent

LONG_TERM_SYSTEM_PROMPT = """\
You have access to memory tools. Use them proactively:

- When the user shares personal information, preferences, or important facts, \
call `save_memory` to store each distinct fact with a descriptive key.
- At the START of every new conversation, call `recall_memories` with a broad \
query like "user" to check for previously stored information. Reference any \
recalled facts naturally in your greeting.
- When a question seems like it could relate to previously stored info, call \
`recall_memories` before answering.

Be thorough: save multiple facts from a single message if it contains several \
pieces of information (e.g., name, profession, and hobbies should each be saved \
separately)."""


def _make_memory_tools():
    """Create save_memory and recall_memories tools for long-term memory."""

    @tool
    def save_memory(
        key: str,
        value: str,
        store: Annotated[Any, InjectedStore()],
    ) -> str:
        """Save a fact about the user for future conversations.

        Use this when the user shares preferences, personal info, or important
        context you should remember later. Use a descriptive key like
        'name', 'allergy', 'favorite_language', etc.
        """
        store.put(("user_memories",), key, {"value": value})
        return f"Saved: {key} = {value}"

    @tool
    def recall_memories(
        query: str,
        store: Annotated[Any, InjectedStore()],
    ) -> str:
        """Recall previously saved facts about the user.

        Call this at the start of new conversations or when context seems
        relevant. The `query` parameter can be useful with semantic memory.
        """
        items = store.search(("user_memories",), limit=20)
        if not items:
            return "No memories found."
        return "\n".join(f"- {item.key}: {item.value['value']}" for item in items)

    return [save_memory, recall_memories]


def make_agent(
    model_str: str = "anthropic:claude-haiku-4-5-20251001",
    system_prompt: str | None = None,
    memory_type: str = "none",
):
    """Create a deep agent with the specified model provider and memory type.

    Args:
        model_str: Provider and model in "provider:model" format.
                   Examples: "openai:gpt-4o", "anthropic:claude-haiku-4-5-20251001",
                   "google_genai:gemini-2.5-flash"
        system_prompt: Optional system prompt override.
        memory_type: One of "none", "session", or "long_term".
            - "none": No memory persistence (default, original behavior).
            - "session": Checkpointer-based session memory. Persists conversation
              state by thread_id within a single process.
            - "long_term": Store-based long-term memory with explicit save/recall
              tools. Facts persist across threads.

    Returns:
        A compiled LangGraph agent supporting .invoke(), .stream(), .astream().
    """
    if memory_type not in ("none", "session", "long_term"):
        raise ValueError(f"Unknown memory_type: {memory_type!r}")

    model = init_chat_model(model_str)
    kwargs: dict[str, Any] = {}

    if system_prompt:
        kwargs["system_prompt"] = system_prompt

    if memory_type == "session":
        kwargs["checkpointer"] = MemorySaver() # In production, this may look like: checkpointer = PostgresSaver.from_conn_string(DB_URI)

    elif memory_type == "long_term":
        kwargs["checkpointer"] = MemorySaver()
        kwargs["store"] = InMemoryStore()      # In production, this may look like: store = PostgresStore.from_conn_string(DB_URI)
        kwargs["tools"] = _make_memory_tools()
        # Prepend long-term memory instructions to any user-provided prompt
        base_prompt = system_prompt or ""
        combined = (LONG_TERM_SYSTEM_PROMPT + "\n\n" + base_prompt).strip()
        kwargs["system_prompt"] = combined

    return create_deep_agent(model=model, **kwargs)
