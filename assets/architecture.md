# Diagrams

These Mermaid diagrams render natively on GitHub. For the Medium article, open the pipeline
block at <https://mermaid.live> and export a PNG (Actions > Export), so the repo and the
article share the exact same diagram.

## TokenGate pipeline

```mermaid
flowchart LR
    C([Retrieved chunks]):::io
    Q([Query]):::io

    subgraph S1 [Dedup and Rank]
      direction TB
      D[Exact dedup]:::n --> E[BGE-M3 embed<br/>and hybrid rank]:::n --> R[Cross-encoder<br/>rerank]:::n
    end

    subgraph S2 [Select]
      direction TB
      AC[Adaptive cutoff]:::n --> SD[Semantic dedup]:::n --> MM[MMR diversity]:::n
    end

    subgraph S3 [Budget]
      direction TB
      B[Value-per-token<br/>budget]:::n
      CMP[Compression<br/>off by default]:::off
    end

    C --> D
    Q --> E
    R --> AC
    MM --> B
    CMP -.-> B
    B --> P([Optimized prompt]):::io --> LLM([Any LLM]):::io
    B --> A[(Audit<br/>kept / dropped / why)]:::audit

    classDef io fill:#eef2ff,stroke:#6366f1,color:#3730a3
    classDef n fill:#ffffff,stroke:#cbd2ea,color:#27304a
    classDef audit fill:#ecfdf5,stroke:#10b981,color:#065f46
    classDef off fill:#f6f7fb,stroke:#c4c9da,stroke-dasharray:4 4,color:#8a90a4
    style S1 fill:#fafbff,stroke:#e2e6f5
    style S2 fill:#fafbff,stroke:#e2e6f5
    style S3 fill:#fafbff,stroke:#e2e6f5
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
