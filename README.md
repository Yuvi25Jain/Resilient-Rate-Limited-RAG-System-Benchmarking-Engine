# ⚡ Resilient Rate-Limited RAG System & Benchmarking Engine

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Groq API](https://img.shields.io/badge/LLM-Groq%20%28openai%2Fgpt--oss--120b%29-orange.svg)](https://groq.com/)
[![Vector Store](https://img.shields.io/badge/VectorDB-ChromaDB%20%2B%20BM25-green.svg)](https://docs.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade, fault-tolerant **Retrieval-Augmented Generation (RAG)** engine built to handle strict LLM API rate limits, traffic spikes, and context-window degradation without failing or hallucinating.

---

## 🎯 Architecture Highlights

- **4-Step Agentic Pipeline:** Query Router $\rightarrow$ Hybrid Retrieval (Dense ChromaDB + Sparse BM25 via RRF) $\rightarrow$ LLM Generator $\rightarrow$ Self-Reflection Evaluator.
- **Dynamic Degraded Mode:** Client-side rate-limit monitor dynamically trims context windows by 50% when API usage reaches 80% TPM/RPM, preventing `HTTP 429` failures.
- **Zero-Hallucination Guardrails:** Strict out-of-scope refusal handling and mandatory inline metadata citations (`[Paper, Section, Page]`).
- **Client-Side Exact Caching:** Instant zero-latency ($<1\text{ ms}$) cache responses for repeated queries.
- **Stand-in Simulator & Load Benchmarking:** Fully calibrated mock model service mirroring Groq latency distributions and rate limits across Steady, Burst, and Sustained Overload traffic shapes.

---

## 📚 Ingested Knowledge Base

The system indexes three arXiv research papers with structural metadata chunking:
1. **Attention Is All You Need** (`arXiv:1706.03762v7`)
2. **Moshi: Speech-Text Joint Audio Model** (`arXiv:2410.00037v1`)
3. **Dynamic Chunking for Large Context RAG** (`arXiv:2507.07955v2`)

---

## 📊 Benchmark Results (100 Request Load Test)

### Latency Percentiles (p50 / p95 / p99)

| Scenario | Source | TTFT (p50 / p95 / p99) | Total Latency (p50 / p95 / p99) |
| :--- | :--- | :--- | :--- |
| **Steady** (15 reqs) | Stand-in | 742.8 ms / 1019.8 ms / 1031.2 ms | 1005.9 ms / 1302.3 ms / 1320.1 ms |
| **Burst** (35 reqs) | Stand-in | 0.0 ms / 964.6 ms / 1021.4 ms | 0.0 ms / 1237.3 ms / 1307.1 ms |
| **Overload** (50 reqs)| Stand-in | 0.0 ms / 0.0 ms / 0.0 ms | 0.0 ms / 0.0 ms / 0.0 ms |

### Outcome & Resilience Metrics

| Scenario | Success Rate (%) | Rate-Limited / Refused (%) | Avg Tokens / Answer |
| :--- | :--- | :--- | :--- |
| **Steady** | 100.0% | 0.0% | 24.0 |
| **Burst** | 42.9% | 57.1% | 10.3 |
| **Overload** | 0.0% | 100.0% | 0.0 |

---

## 🛠️ Installation & Setup

```bash
# 1. Clone repository
git clone [https://github.com/Yuvi25Jain/Resilient-Rate-Limited-RAG-System-Benchmarking-Engine.git](https://github.com/Yuvi25Jain/Resilient-Rate-Limited-RAG-System-Benchmarking-Engine.git)
cd Resilient-Rate-Limited-RAG-System-Benchmarking-Engine

# 2. Activate Virtual Environment & Install Dependencies
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e .

# 3. Setup Groq API Key
echo "GROQ_API_KEY=gsk_your_key_here" > .env

# 4. Ingest Papers & Run Streamlit App
python ingest.py
streamlit run app.py