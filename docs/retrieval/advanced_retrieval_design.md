# Advanced Retrieval and Memory Design

This note documents the optional document retrieval pipeline used to strengthen the Copilot memory architecture.

## Pipeline

1. Extract text from uploaded documents.
2. Split text using overlapping chunks.
3. Store metadata such as company, department, topic, document type, chunk number and source file.
4. Filter by metadata before retrieval.
5. Run hybrid search using TF-IDF semantic-style similarity and BM25 keyword relevance.
6. Rerank 50-100 candidates and keep the top 5-10 chunks.
7. Use only retrieved evidence for Copilot answers.
8. Evaluate retrieval quality using precision@k, recall@k, MRR and nDCG@k.

## Dissertation position

The customer churn workflow remains the main implemented and evaluated case study. Advanced retrieval strengthens the memory and future enterprise design. It can be shown as an optional implemented support feature and discussed as future work for document-grounded Copilot answers.
