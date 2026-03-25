# Write-up

## Video Walkthrough

[Loom video walkthrough](https://www.loom.com/share/792ce1a6b73c4fc0bd0c5696a8bd6b60)

## Overview

Starting from the provided barebones CLI agent, I implemented three memory strategies using LangGraph primitives — baseline (no memory), session memory (checkpointer), and long-term memory (store + custom tools) — and built a scripted harness that demonstrates their behavioral differences side-by-side.

The key files added or modified:
- **`core.py`** — Extended `make_agent()` with a `memory_type` parameter that wires up the appropriate LangGraph primitives (`MemorySaver`, `InMemoryStore`, custom tools)
- **`harness.py`** — Runs three scripted conversations against all memory types to show distinct recall behavior
- **`test_memory.py`** — Integration tests verifying each strategy's properties

See [HARNESS.md](HARNESS.md) for detailed harness documentation, memory design dimensions, trade-offs, and further directions.

## Summary of trade-offs

| | Baseline | Session | Long-term |
|---|---|---|---|
| **Persistence** | None | Full graph state by `thread_id` | Selective facts in shared store |
| **Cross-thread recall** | No | No | Yes |
| **Complexity** | Zero | Low (one-line config) | Higher (tools + prompt engineering) |
| **Scalability concern** | N/A | Unbounded state growth | LLM judgment reliability |
| **Production use case** | Stateless bots | Support chats, IDE copilots | Personal assistants, healthcare |
