# crusader
# Agentic Peer-Review 2.0: System Architecture Design

## Executive Summary

Agentic Peer-Review 2.0 (APR-2.0) proposes a paradigm shift in scholarly publishing, replacing traditional, human-intensive peer review with a fully automated, end-to-end framework. This system leverages a mesh of narrowly scoped, cooperating AI agents orchestrated by an event-driven Directed Acyclic Graph (DAG). APR-2.0 aims to address the unsustainable volume of research output, combat quality control failures like plagiarism and data fabrication, and enhance the transparency and auditability of the peer-review process. By employing fine-tuned small language models (SLMs) for specialized tasks, the system seeks to deliver higher quality, faster reviews, and more consistent editorial decisions. Key features include robust security, comprehensive explainability with cited evidence, version control for reproducibility, and seamless interoperability with existing scholarly infrastructure. This document outlines the architecture, agent roster, data contracts, a 3-month MVP roadmap, and risk mitigation strategies for APR-2.0, paving the way for a more efficient, trustworthy, and scalable future for scholarly communication.

---

## Detailed Architecture

The APR-2.0 system processes manuscript submissions through a pipeline of specialized agents, from initial intake and validation to comprehensive review, ethical scrutiny, decision-making, and finally, production and post-publication monitoring. The architecture is built upon a microservices model, with agents communicating via an event bus and orchestrated by a LangGraph-based DAG.

### Intake → Publish Pipeline Stages:

1.  **Submission Intake:** Authors upload manuscripts and metadata via a secure portal.
2.  **Pre-Flight Checks:** Initial validation of file formats, metadata completeness, and basic structural integrity.
3.  **Ethical Screening:** Checks for plagiarism, image manipulation, conflicts of interest disclosures, and IRB/ethics committee approvals.
4.  **Content Triage & Scoping:** Determines the subject area, novelty, and potential impact to route for appropriate specialized review.
5.  **Technical Review (Iterative):** Agents assess methodology, statistical validity, data integrity, reproducibility, and clarity. This stage can involve multiple specialized agents.
6.  **Synthesis & Recommendation:** An aggregator agent compiles findings from all review agents, generating a summary report and a provisional recommendation.
7.  **Author Rebuttal Loop (Conditional):** If revisions are requested, authors submit a revised manuscript and a point-by-point response. The relevant agents re-evaluate the changes.
8.  **Editorial Decision Gate:** An editor (human-in-the-loop) reviews the aggregated report, agent verdicts, and author responses. They can override, request further clarification, or approve the recommendation. Policy gates ensure compliance.
9.  **Production & Dissemination:** Upon acceptance, agents handle JATS XML conversion, Crossref DOI registration, ORCID updates, and packaging for publication.
10. **Post-Publication Monitoring:** Agents monitor citations, discussions, and errata, flagging potential issues or updates.

### Orchestration Graph (Mermaid DAG)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'lineColor': '#555', 'primaryColor': '#EFF2FA', 'primaryTextColor': '#333', 'primaryBorderColor': '#A9B1D6', 'secondaryColor': '#E0E0E0', 'tertiaryColor': '#F0F0F0'}}}%%
graph TD
    A[Manuscript Submission API] --> B(Submission Intake Agent);
    B --> C{Pre-Flight Check Agent};
    C -- Valid --> D(Plagiarism Detection Agent);
    C -- Invalid --> X1(Notify Author: Format Error);
    D -- Clear --> E(Ethics & Compliance Agent);
    D -- Flagged --> X2(Notify Editor: Plagiarism);
    E -- Clear --> F(Content Triage Agent);
    E -- Concerns --> X3(Notify Editor: Ethical Concerns);
    F --> G(Assign Reviewer Agents);
    G --> H1(Methodology Review Agent);
    G --> H2(Statistical Review Agent);
    G --> H3(Data Integrity Agent);
    G --> H4(Reproducibility Agent);
    G --> H5(Clarity & Language Agent);
    H1 --> I(Review Aggregator Agent);
    H2 --> I;
    H3 --> I;
    H4 --> I;
    H5 --> I;
    I --> J{Provisional Decision};
    J -- Accept/Minor Revise --> K(Author Notification Agent: Decision);
    J -- Major Revise --> L{Author Rebuttal Loop};
    J -- Reject --> M(Author Notification Agent: Reject);
    L -- Revised Manuscript --> B;
    K -- Acceptance --> N(JATS XML Conversion Agent);
    N --> O(Crossref DOI Agent);
    O --> P(ORCID Update Agent);
    P --> Q(Publication Package Agent);
    Q --> R(Archive & Disseminate);
    R --> S(Post-Publication Monitoring Agent);
    subgraph "Human-in-the-Loop Gates"
        X2 --> E_Override1(Editor Review & Override);
        X3 --> E_Override2(Editor Review & Override);
        I --> E_Override3(Editor Review: Synthesis & Recommendation);
        E_Override3 -- Approve --> K;
        E_Override3 -- Request Clarification --> G;
        E_Override3 -- Reject --> M;
    end
    subgraph "Security & Logging"
        B --> SL1(Provenance Logger);
        C --> SL1;
        D --> SL1;
        E --> SL1;
        F --> SL1;
        H1 --> SL1;
        H2 --> SL1;
        H3 --> SL1;
        H4 --> SL1;
        H5 --> SL1;
        I --> SL1;
        N --> SL1;
        O --> SL1;
        P --> SL1;
        Q --> SL1;
    end

    classDef agent fill:#C9D1F8,stroke:#5C6BC0,stroke-width:2px;
    classDef decision fill:#FFDDAA,stroke:#FFA500,stroke-width:2px;
    classDef human_gate fill:#D4E8D4,stroke:#38761D,stroke-width:2px;
    classDef io fill:#E0E0E0,stroke:#757575,stroke-width:2px;
    classDef notify fill:#FFCDD2,stroke:#D32F2F,stroke-width:2px;
    classDef process fill:#B3E5FC,stroke:#0288D1,stroke-width:2px;

    class A,R,X1,X2,X3,M,K io;
    class B,C,D,E,F,G,H1,H2,H3,H4,H5,I,N,O,P,Q,S,SL1 agent;
    class J,L decision;
    class E_Override1,E_Override2,E_Override3 human_gate;

    
