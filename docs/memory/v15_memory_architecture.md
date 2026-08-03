# v15 Memory Architecture

## Short-term memory

Implemented through application session state. It stores the active dataset, selected target column, last user question, last Copilot answer, latest topic/column context and chat history. This supports follow-up questions during one session.

## Structured long-term memory

PostgreSQL is used in the scalable architecture for structured business data, dataset metadata, readiness gates, model runs, metrics, predictions and audit events. The local prototype can continue to run without PostgreSQL by using in-memory/session state and local export files.

## Semantic long-term memory

ChromaDB is used as the semantic memory layer. It stores embeddings of dataset summaries, readiness reports, explanations, reports, Copilot answers and generated insights. The prototype includes a JSONL fallback so that it works even when ChromaDB is not installed.

## MongoDB decision

MongoDB is removed from the core architecture. It was originally considered for document/chat storage, but ChromaDB provides a stronger justification for retrieval-based AI and grounded Copilot responses.
