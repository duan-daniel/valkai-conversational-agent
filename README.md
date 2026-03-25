# take-home

CLI chat agent built on [LangChain Deep Agents](https://github.com/langchain-ai/deepagents) with **three memory strategies** that demonstrate how different persistence approaches affect conversational context and recall.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- At least one LLM provider API key

## Quick start

```bash
git clone https://github.com/valkai-tech/take-home.git
cd take-home
uv sync
cp .env.example .env
# Fill in your API key(s) in .env
```

## Usage

```bash
# Default — no memory
uv run chat

# With session memory (remembers within this session's thread)
uv run chat --memory session

# With long-term memory (saves/recalls facts across threads)
uv run chat --memory long_term

# Custom model + system prompt
uv run chat --model openai:gpt-4o --system "You are a pirate."
```

Type `quit` or `exit` to end the session.

## Running the harness

The harness is a scripted conversation that runs all three memory types side-by-side and prints the results, making behavioral differences immediately visible.

```bash
uv run harness
```

See **[HARNESS.md](HARNESS.md)** for full documentation including:
- How each memory type works and their LangGraph primitives
- Memory design dimensions (what/when/how) and trade-offs
- The design space as a 2x2 matrix (storage strategy vs. scope)
- Real-world industry context for each memory type
- Production persistence (swapping to Postgres)
- Further directions: semantic memory with vector embeddings

## Running evals

```bash
# All tests
uv run pytest evals/ -v

# Original agent tests only
uv run pytest evals/test_agent.py -v

# Memory-specific tests only
uv run pytest evals/test_memory.py -v
```

Evals make real LLM calls (not mocked) to verify behavior end-to-end.

## Supported providers

| Provider  | Model string example                          | Required env var       |
|-----------|-----------------------------------------------|------------------------|
| Anthropic | `anthropic:claude-haiku-4-5-20251001` (default) | `ANTHROPIC_API_KEY`    |
| OpenAI    | `openai:gpt-4o`                               | `OPENAI_API_KEY`       |
| Google    | `google_genai:gemini-2.5-flash`               | `GOOGLE_API_KEY`       |

Any model supported by LangChain's [`init_chat_model`](https://docs.langchain.com/oss/python/langchain/models) works — just pass the `provider:model` string.

## Project structure

```
take-home/
├── pyproject.toml          # uv project config, dependencies, scripts
├── .env.example            # API key template
├── HARNESS.md              # Harness docs, memory analysis, and trade-offs
├── src/
│   └── agent/
│       ├── core.py         # Agent factory with memory_type support
│       ├── cli.py          # Interactive chat REPL
│       └── harness.py      # Scripted memory comparison harness
└── evals/
    ├── test_agent.py       # Barebones agent evals
    └── test_memory.py      # Memory-specific evals
```
