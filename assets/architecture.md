# Diagrams

These render on GitHub as-is. To export a PNG or SVG for a Medium article or a LinkedIn
post, paste the code block into <https://mermaid.live> and use Actions > Export.

## TokenGate pipeline

```mermaid
flowchart LR
    C[Retrieved chunks] --> D[Exact dedup]
    Q[Query] --> E
    D --> E[BGE-M3 embed<br/>hybrid rank]
    E --> R[BGE cross-encoder<br/>rerank]
    R --> AC[Adaptive relevance<br/>cutoff]
    AC --> SD[Semantic dedup]
    SD --> M[MMR diversity]
    M --> B[Value-per-token<br/>budget]
    B --> P[Optimized prompt]
    B --> A[(Audit report:<br/>kept / dropped / why)]
    P --> LLM[Any LLM]

    CMP[Compression<br/>opt-in, off by default]:::off -.-> B
    classDef off fill:#f4f4f8,stroke:#b9b9c9,stroke-dasharray:5 5,color:#6b7180
```

## Beacon system (local RAG app using TokenGate)

```mermaid
flowchart TD
    F[Your folders] --> SC[Scan + extract text<br/>TXT / MD / PDF / DOCX / images]
    SC --> CH[Token-aware chunking]
    CH --> EMB[BGE-M3 embeddings]
    EMB --> DB[(LanceDB<br/>local vector store)]

    Q[Your question] --> RET[Retrieve top-50]
    DB --> RET
    RET --> GATE{Relevance gate}
    GATE -->|on-topic| TG[TokenGate.optimize]
    GATE -->|on-topic| BL[Baseline RAG<br/>rerank + top-N + stuff]
    TG --> OLL[Ollama local LLM]
    BL --> OLL
    OLL --> ANS[Streamed answer<br/>+ citations + full audit]

    classDef store fill:#eef2ff,stroke:#4f46e5,color:#312e81
    class DB store
```
