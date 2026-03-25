# Memory Comparison Harness

## Running the harness

The harness is a scripted conversation that runs all three memory types side-by-side and prints the results, making behavioral differences immediately visible.

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

| Memory Type | Conv B (new thread) | Conv C (resume thread A) |
|-------------|---------------------|--------------------------|
| `none`      | No recall           | No recall                |
| `session`   | No recall           | Full recall              |
| `long_term` | Recalls stored facts | Full recall              |

---

## Memory types

### 1. Baseline (`none`) — No memory

The default. Each invocation is independent. The caller passes a message list, but nothing is persisted between sessions. This is the original behavior.

**LangGraph primitive:** None — the caller manages message history manually.

### 2. Session memory (`session`) — Checkpointer

Uses LangGraph's `MemorySaver` (an in-memory checkpointer) to persist the full conversation state by `thread_id`. All turns within the same thread are automatically remembered. A new thread starts from scratch.

**LangGraph primitive:** `checkpointer=MemorySaver()` passed to `create_deep_agent()`.

**How it differs from baseline:** Deep Agents already include a built-in transient filesystem in their graph state, and the baseline CLI manually accumulates messages across turns — so within a single CLI session, baseline and session memory behave identically from the user's perspective. The difference is **who is responsible for state persistence**. With baseline, the caller must manage the message list and would need to build its own save/restore infrastructure to resume a conversation later. With session memory, the checkpointer handles this automatically — any code that knows the `thread_id` can resume the conversation without needing to reconstruct the message history. In the harness, this shows up in Conversation C: session memory seamlessly resumes thread "conv-a" via the checkpointer, while baseline starts fresh because no one saved the messages from Conversation A.

### 3. Long-term memory (`long_term`) — Store + Tools

Uses an `InMemoryStore` alongside a checkpointer. The agent has two tools — `save_memory(key, value)` and `recall_memories(query)` — that let it explicitly store and retrieve facts. A system prompt instructs the agent to save important user information and recall it in new conversations.

Facts persist **across threads** because the store is shared, while the checkpointer handles within-thread continuity.

**LangGraph primitives:** `checkpointer=MemorySaver()`, `store=InMemoryStore()`, and `tools=[save_memory, recall_memories]` passed to `create_deep_agent()`.

---

## Memory design dimensions

A few key terms used throughout this section (these are general conversational AI concepts, not LangChain-specific — though LangGraph implements threads via its `thread_id` config parameter):

- **Turn** — A single user message + agent response pair.
- **Thread** (or **session**) — A sequence of turns representing one continuous conversation (e.g., a support chat from open to close). In LangGraph, each thread is identified by a `thread_id`.

When designing memory for AI agents, there are several choices to make: **what** information is stored, **when** it is stored, and **how** it is stored.

| Dimension | Baseline | Session Memory | Long-term Memory |
|-----------|----------|----------------|------------------|
| **What** is stored | Nothing | Full message history (all turns) | Extracted facts & preferences (structured key-value pairs) |
| **When** is it stored | N/A | Automatically after every turn | Agent decides via tool calls when info seems important |
| **How** is it stored | N/A | Serialized graph state keyed by `thread_id` | Key-value store with namespace hierarchy, queryable across threads |

### Trade-offs by dimension

**What is stored:**
- Storing everything (session) is simple but grows unbounded and includes noise — every "um" and tangent is preserved.
- Storing extracted facts (long-term) is selective but relies on the LLM's judgment about what matters. Important details can be missed; irrelevant ones can be saved.

**When it is stored:**
- Automatic storage (session) requires no prompt engineering and never fails to capture data, but offers no control over what gets kept.
- Agent-driven storage (long-term) is flexible but can be unreliable — the agent might forget to save, save too aggressively, or save redundant entries. System prompt design is critical.

**How it is stored:**
- Thread-scoped storage (session) is simple and naturally isolated, but knowledge is siloed — nothing learned in one conversation carries to the next.
- Shared namespace storage (long-term) enables cross-conversation recall but requires careful key/namespace design to avoid collisions and support efficient retrieval at scale.

### The design space is a matrix, not a fixed pairing

In this project, we paired session memory with "store everything" and long-term memory with "selectively store." But storage strategy (what/when) and scope (thread vs. cross-thread) are independent design choices — you could mix and match:

|  | Thread-scoped | Cross-thread |
|--|---------------|--------------|
| **Store everything** | Session memory (this project) | Possible — auto-dump full transcripts into a shared store. But expensive to search and mostly duplicates what a checkpointer does. |
| **Store selectively** | Possible — summarize or prune conversation history before checkpointing (e.g., deepagents' `SummarizationMiddleware`). | Long-term memory (this project) |

We chose the diagonal because it reflects the most common production pattern:
- Session memory stores everything because within a single thread, the full context is usually small enough and all of it is relevant.
- Long-term memory stores selectively because across many conversations, you'd accumulate too much noise storing everything — selective extraction keeps the store compact and retrieval fast.

The other combinations are valid but less common. Session + selective is essentially context window management (pruning/compacting), which this project intentionally avoids per the spec. Long-term + store everything would work but gets expensive quickly — you'd need to search through massive raw transcripts to find relevant info in a new thread.

---

## Real-world industry context

### Baseline (No Memory)
Stateless API endpoints, one-shot Q&A bots, search assistants. Every request is independent. Example: a Slack bot that answers questions from documentation — no need to remember prior queries.

### Session Memory (Checkpointer)
A "thread" maps to a bounded conversation session. In production:
- **Customer support chat** — The user opens a ticket, has a multi-turn conversation, and closes it. The `thread_id` is the ticket/case ID. A new ticket starts fresh.
- **IDE copilot** — Each coding session or file context is a thread. The agent remembers what you've been working on this session but doesn't carry yesterday's context.
- **Meeting assistant** — Each meeting is a thread. The agent tracks action items and discussion within that meeting.

### Long-term Memory (Store)
Facts and preferences persist across all conversations. In production:
- **Personal AI assistant** — Learns your role, team, preferences, and communication style over weeks of daily use. Across departments and roles, the agent adapts to each user individually.
- **Healthcare chatbot** — Remembers patient allergies, medications, and history across separate appointments.
- **Enterprise knowledge worker** — Remembers that "the Q3 report uses metric X" so you don't re-explain context every Monday.
- **Cross-user pattern learning** — At scale, aggregated long-term memory across users can surface organizational patterns (e.g., "engineering teams frequently ask about deployment configs").

### Production persistence

In this demo, both `MemorySaver` and `InMemoryStore` store data in Python dictionaries — everything is lost when the process exits. In production, you'd swap them for Postgres-backed equivalents with an identical interface:

```python
# In-memory (this project)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
checkpointer = MemorySaver()
store = InMemoryStore()

# Production (same interface, persistent storage)
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
DB_URI = "postgresql://user:pass@localhost:5432/mydb"
checkpointer = PostgresSaver.from_conn_string(DB_URI)
checkpointer.setup()  # creates tables on first run
store = PostgresStore.from_conn_string(DB_URI)
store.setup()
```

Both are passed to `create_deep_agent()` the same way — the agent code doesn't change at all. The checkpointer and store write to separate tables: checkpoints store full serialized graph state keyed by `(thread_id, checkpoint_id)`, while the store holds explicit key-value pairs keyed by `(namespace, key)`.

---

## Further directions for a 4th Memory Type: Semantic Memory

Our long-term memory's `recall_memories` tool currently lists **all** stored facts with `store.search(("user_memories",), limit=20)`. This works fine with a handful of facts, but breaks down at scale — if the agent has stored hundreds of facts across weeks of conversations, dumping them all into the prompt is expensive and noisy.

**Semantic memory** solves this by retrieving only the facts that are **relevant to the current query**, using vector similarity search.

### What are vector embeddings?

An embedding model (like OpenAI's `text-embedding-3-small`) converts text into a list of numbers — a **vector** — with hundreds or thousands of dimensions:

```
"I'm allergic to shellfish"  →  [0.023, -0.041, 0.087, 0.012, ...]  (1536 numbers)
"What should I avoid eating?" →  [0.019, -0.038, 0.091, 0.008, ...]  (1536 numbers)
```

Texts with **similar meaning** produce vectors that are **close together** in this high-dimensional space, even if they share few words. The distance between vectors is measured using cosine similarity — a score from 0 (unrelated) to 1 (identical meaning).

This means you can search by meaning: "what should I avoid at a seafood restaurant?" would match the stored fact about shellfish allergy, even though the query doesn't contain the word "shellfish" or "allergy."

### How it differs from our long-term memory

| | Long-term (implemented) | Semantic (not implemented) |
|---|---|---|
| **Storage** | `store.put(namespace, key, {"value": text})` | Same, but store is initialized with an embedding index |
| **Retrieval** | `store.search(namespace, limit=20)` — lists all items | `store.search(namespace, query="natural language", limit=5)` — ranked by meaning similarity |
| **Scales to** | Tens of facts (dumps all into prompt) | Hundreds or thousands of facts (retrieves only relevant ones) |

### Sample implementation

LangGraph's `InMemoryStore` already supports vector search — you just configure an embedding index:

```python
from langchain_openai import OpenAIEmbeddings
from langgraph.store.memory import InMemoryStore

# Initialize store with vector index
store = InMemoryStore(index={
    "dims": 1536,                                          # embedding dimensions
    "embed": OpenAIEmbeddings(model="text-embedding-3-small"),  # embedding model
    "fields": ["value"],                                   # which fields to embed
})

# Storing works the same way
store.put(("user_memories",), "allergy", {"value": "allergic to shellfish"})

# But searching now uses vector similarity
results = store.search(
    ("user_memories",),
    query="what should I avoid at a seafood restaurant?",  # similarity search
    limit=5,
)
for item in results:
    print(f"{item.key}: {item.value['value']} (score: {item.score:.3f})")
    # allergy: allergic to shellfish (score: 0.847)
```

The `recall_memories` tool would change by one line — passing `query` to `store.search()`:

```python
# Long-term (current) — lists everything
items = store.search(("user_memories",), limit=20)

# Semantic — searches by meaning
items = store.search(("user_memories",), query=query, limit=5)
```

### Why we didn't implement it

Semantic memory requires an **embedding API** to convert text to vectors. Anthropic doesn't offer an embeddings model, so this would require an additional API key (OpenAI or Google).
