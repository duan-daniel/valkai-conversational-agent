# Memory Comparison Harness

## Running the harness

The harness is a scripted conversation that runs all three memory strategies side-by-side and prints the results, making behavioral differences immediately visible.

```bash
uv run harness
```

## How it works

The harness runs **three conversations** against each memory type (`none`, `session`, `long_term`):

1. **Conversation A** — The user shares personal info (name, hobbies, allergy) in thread `conv-a`
2. **Conversation B** — A new thread (`conv-b`) tests cross-thread recall
3. **Conversation C** — Resumes thread `conv-a` to test session resumption

After the conversations, the harness inspects **what each memory type actually stored** under the hood — showing the difference between storing everything verbatim (session) vs. selectively extracting facts (long-term).

### Expected results

| Memory Strategy | Conv B (new thread) | Conv C (resume thread A) |
|-------------|---------------------|--------------------------|
| `none`      | No recall           | No recall                |
| `session`   | No recall           | Full recall              |
| `long_term` | Recalls stored facts | Full recall              |

---

## Memory strategies

### 1. Baseline (`none`) — No memory

The default. Each invocation is independent. The caller passes a message list, but nothing is persisted between sessions. This is the original behavior.

**LangGraph primitive:** None — the caller manages message history manually.

### 2. Session memory (`session`) — Checkpointer

Uses LangGraph's `MemorySaver` (an in-memory checkpointer) to persist the full conversation state by `thread_id`. All turns within the same thread are automatically remembered. A new thread starts from scratch.

**LangGraph primitive:** `checkpointer=MemorySaver()` passed to `create_deep_agent()`.

**How it differs from baseline:** Within a single CLI session, baseline and session memory behave identically from the user's perspective — the difference is **who manages the state**. With baseline, the caller must build its own save/restore infrastructure. With session memory, the checkpointer handles it automatically — any code that knows the `thread_id` can resume the conversation. In the harness, this shows up in Conversation C: session memory seamlessly resumes thread "conv-a" while baseline starts fresh.

### 3. Long-term memory (`long_term`) — Store + Tools

Uses an `InMemoryStore` alongside a checkpointer. The agent has two tools — `save_memory(key, value)` and `recall_memories(query)` — that let it explicitly store and retrieve facts. A system prompt instructs the agent to save important user information and recall it in new conversations.

Facts persist **across threads** because the store is shared, while the checkpointer handles within-thread continuity.

**LangGraph primitives:** `checkpointer=MemorySaver()`, `store=InMemoryStore()`, and `tools=[save_memory, recall_memories]` passed to `create_deep_agent()`.

### 4. Semantic memory (not implemented) — Store + Embeddings

Extends long-term memory with vector embeddings so that recall retrieves only facts **relevant to the current query** by meaning similarity, rather than listing everything. This is critical for scaling beyond a handful of stored facts. See [further details below](#further-directions-semantic-memory).

---

## Memory design dimensions

When designing memory for AI agents, there are three key dimensions: **what** is stored, **when** it is stored, and **how** it is stored.

| Dimension | Baseline | Session Memory | Long-term Memory |
|-----------|----------|----------------|------------------|
| **What** is stored | Nothing | Full message history (all turns) | Extracted facts & preferences (structured key-value pairs) |
| **When** is it stored | N/A | Automatically after every turn | Agent decides via tool calls when info seems important |
| **How** is it stored | N/A | Serialized graph state keyed by `thread_id` | Key-value store with namespace hierarchy, queryable across threads |

### Trade-offs

- **Baseline** stores nothing — zero overhead and fully stateless, but the caller must manage all context manually and nothing persists between sessions.
- **Session memory** stores everything automatically — simple and reliable, but grows unbounded and includes noise.
- **Long-term memory** stores selectively via agent tool calls — more efficient, but relies on the LLM's judgment about what matters. System prompt design is critical to make tool usage reliable.

### The design space is a matrix

Storage strategy (what/when) and scope (thread vs. cross-thread) are independent axes:

|  | Thread-scoped | Cross-thread |
|--|---------------|--------------|
| **Store everything** | Session memory (this project) | Auto-dump transcripts into shared store |
| **Store selectively** | Summarize/prune history before checkpointing | Long-term memory (this project) |

We chose the diagonal because it reflects the most common production pattern. The other combinations are valid but less common.

---

## Real-world industry context

- **Baseline** — Stateless API endpoints, one-shot Q&A bots, search assistants.
- **Session memory** — Customer support chats (each ticket is a thread), IDE copilots (each coding session is a thread), meeting assistants.
- **Long-term memory** — Personal AI assistants that learn preferences over weeks, healthcare chatbots that remember patient allergies across appointments, enterprise knowledge workers that retain context across sessions.

### Production persistence

In this demo, both `MemorySaver` and `InMemoryStore` are in-memory — everything is lost when the process exits. In production, you'd swap them for `PostgresSaver` and `PostgresStore`. The interface is identical — the agent code doesn't change at all.

```python
# Production (same interface, persistent storage)
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
DB_URI = "postgresql://user:pass@localhost:5432/mydb"
checkpointer = PostgresSaver.from_conn_string(DB_URI)
store = PostgresStore.from_conn_string(DB_URI)
```

---

## Further directions: Semantic Memory

Our long-term memory lists **all** stored facts on recall. This works with a handful of facts but breaks down at scale. **Semantic memory** uses vector embeddings to retrieve only facts **relevant to the current query** by meaning similarity.

For example, asking "what should I avoid at a seafood restaurant?" would match the stored shellfish allergy fact, even without the word "allergy" in the query.

### How it differs from our long-term memory

| | Long-term (implemented) | Semantic (not implemented) |
|---|---|---|
| **Retrieval** | `store.search(namespace, limit=20)` — lists all items | `store.search(namespace, query="...", limit=5)` — ranked by similarity |
| **Scales to** | Tens of facts | Hundreds or thousands of facts |

### Implementation

LangGraph's `InMemoryStore` already supports vector search — you just configure an embedding index:

```python
store = InMemoryStore(index={
    "dims": 1536,
    "embed": OpenAIEmbeddings(model="text-embedding-3-small"),
    "fields": ["value"],
})
```

The `recall_memories` tool would change by one line — passing `query` to `store.search()`.

### Why we didn't implement it

Semantic memory requires an embedding API. Anthropic doesn't offer one, so this would require an additional API key (OpenAI or Google).
