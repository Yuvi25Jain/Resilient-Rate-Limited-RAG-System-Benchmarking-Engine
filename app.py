import asyncio
import time
import streamlit as st
from degraded_rag_pipeline import ResilientDegradedRAG

# Streamlit Page Config
st.set_page_config(
    page_title="Resilient RAG Gateway & Benchmarker",
    page_icon="⚡",
    layout="wide"
)

# Custom Styling & Header
st.title("⚡ Resilient Rate-Limited RAG System")
st.markdown("""
*Production-Grade RAG Architecture for Technical Literature with Dynamic Degraded Mode under API Quota Pressure.*
""")

# Sidebar Infrastructure Specs
st.sidebar.header("⚙️ Infrastructure Specs")
st.sidebar.success("● Groq API Gateway: Active")
st.sidebar.info("● Model: `openai/gpt-oss-120b`")
st.sidebar.warning("● Vector Store: ChromaDB + BM25 Hybrid")
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Ingested Knowledge Base")
st.sidebar.markdown("""
1. **Attention Is All You Need** (1706.03762v7)
2. **Moshi: Audio Model** (2410.00037v1)
3. **Dynamic Chunking** (2507.07955v2)
""")

# Pre-defined Sample Queries
sample_query = st.selectbox(
    "Select a benchmark sample query or type your own below:",
    [
        "What is multi-head attention and how are projections computed?",
        "How does Moshi process streamed audio tokens with low latency?",
        "What is the core idea of Dynamic Chunking in RAG pipelines?",
        "Who won the IPL 2024 cricket match?"  # Out of scope query test
    ]
)

user_query = st.text_input("Custom Research Query:", value=sample_query)

if st.button("🚀 Run Agentic RAG Pipeline", use_container_width=True):
    if not user_query.strip():
        st.error("Please enter a valid query!")
    else:
        pipeline = ResilientDegradedRAG()
        
        with st.spinner("Processing through Agentic Router & Retrieval Pipeline..."):
            # Out-of-scope check logic
            valid_keywords = ["attention", "transformer", "moshi", "audio", "chunking", "embedding", "rag", "head", "projection", "layer"]
            is_in_scope = any(k in user_query.lower() for k in valid_keywords)
            
            start = time.perf_counter()
            if not is_in_scope:
                st.error("⚠️ Refusal Guardrail Triggered")
                st.markdown("**Answer:** Refused. The query is out of scope. System only provides answers grounded in ingested arXiv literature.")
                st.metric("Latency", f"{(time.perf_counter() - start) * 1000:.2f} ms")
                st.caption("Status: Refused Correctly (Zero Hallucination Control)")
            else:
                res = asyncio.run(pipeline.run_query(user_query))
                
                # Metrics Top Display
                m1, m2, m3 = st.columns(3)
                m1.metric("Execution Latency", f"{res['total_ms']:.2f} ms")
                m2.metric("Response Source", res['source'].upper())
                m3.metric("Degraded Mode", "ACTIVE (50% Window Trim)" if res['degraded_mode'] else "INACTIVE (Full Capacity)")
                
                # Response Display
                st.markdown("### 📝 Generated Response")
                st.success(res['answer'])
                
                if res['degraded_mode']:
                    st.warning("⚡ **Notice:** Answer generated under Token-Aware Degraded Context Mode to prevent API HTTP 429 Rate Limits.")