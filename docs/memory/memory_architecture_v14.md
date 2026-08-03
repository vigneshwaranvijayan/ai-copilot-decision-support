# v14 Memory Architecture

## Short-term memory

Implemented using application session state. It stores the active dataset, selected target, latest question, latest answer, last topic/column context and current chat history. This supports follow-up questions during the same session.

## Structured long-term memory

PostgreSQL is the scalable design choice for structured business data, dataset metadata, readiness reports, model runs, metrics and audit records. SQLite is used as the local prototype fallback for audit events.

## Semantic long-term memory

ChromaDB is the revised semantic memory layer. It stores embeddings of dataset summaries, readiness reports, explanations, audit records, generated insights and Copilot answers. The Copilot can retrieve related prior evidence before answering. If ChromaDB is not installed, the prototype uses a JSONL fallback so the app still runs.

## Why MongoDB was removed

MongoDB was originally considered for unstructured records. After supervisor feedback, it was removed from the core design because ChromaDB is a better fit for retrieval-based AI memory and evidence grounding.
