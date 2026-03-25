"""Integration tests for memory types.

These tests make real LLM calls. Requires a valid API key in .env.

Run:
    uv run pytest evals/test_memory.py -v
"""

import pytest
from dotenv import load_dotenv

from agent.core import make_agent

load_dotenv()


# ── Session memory ────────────────────────────────────────────────────────────


@pytest.fixture
def session_agent():
    return make_agent(memory_type="session")


def test_session_memory_within_thread(session_agent):
    """Session memory recalls info within the same thread."""
    config = {"configurable": {"thread_id": "test-same"}}

    session_agent.invoke(
        {"messages": [{"role": "user", "content": "My name is Alice."}]},
        config=config,
    )
    result = session_agent.invoke(
        {"messages": [{"role": "user", "content": "What is my name?"}]},
        config=config,
    )
    assert "Alice" in result["messages"][-1].content


def test_session_memory_forgets_across_threads(session_agent):
    """Session memory does not recall info from a different thread."""
    config_a = {"configurable": {"thread_id": "test-a"}}
    config_b = {"configurable": {"thread_id": "test-b"}}

    session_agent.invoke(
        {"messages": [{"role": "user", "content": "My name is Alice."}]},
        config=config_a,
    )
    result = session_agent.invoke(
        {"messages": [{"role": "user", "content": "What is my name?"}]},
        config=config_b,
    )
    assert "Alice" not in result["messages"][-1].content


# ── Long-term memory ─────────────────────────────────────────────────────────


@pytest.fixture
def long_term_agent():
    return make_agent(memory_type="long_term")


def test_long_term_memory_cross_thread(long_term_agent):
    """Long-term memory recalls stored facts from a different thread."""
    config_a = {"configurable": {"thread_id": "lt-a"}}
    config_b = {"configurable": {"thread_id": "lt-b"}}

    long_term_agent.invoke(
        {"messages": [{"role": "user", "content": "My name is Alice and I love Python."}]},
        config=config_a,
    )
    result = long_term_agent.invoke(
        {"messages": [{"role": "user", "content": "What is my name?"}]},
        config=config_b,
    )
    assert "Alice" in result["messages"][-1].content


# ── Baseline ──────────────────────────────────────────────────────────────────


@pytest.fixture
def baseline_agent():
    return make_agent(memory_type="none")


def test_baseline_no_cross_invocation_memory(baseline_agent):
    """Baseline agent forgets between separate invocations."""
    baseline_agent.invoke(
        {"messages": [{"role": "user", "content": "My name is Alice."}]}
    )
    result = baseline_agent.invoke(
        {"messages": [{"role": "user", "content": "What is my name?"}]}
    )
    assert "Alice" not in result["messages"][-1].content
