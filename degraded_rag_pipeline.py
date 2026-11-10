import os
import time
import asyncio
import hashlib
from typing import Dict, Any, List, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

# --- 1. CLIENT-SIDE EXACT MATCH CACHE ---
class ExactResponseCache:
    def __init__(self):
        self._cache: Dict[str, str] = {}

    def _make_key(self, query: str, context_chunks: List[str]) -> str:
        raw = f"{query.strip().lower()}::" + "::".join(context_chunks)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, query: str, context_chunks: List[str]) -> Optional[str]:
        return self._cache.get(self._make_key(query, context_chunks))

    def set(self, query: str, context_chunks: List[str], response: str):
        self._cache[self._make_key(query, context_chunks)] = response


# --- 2. RESILIENT RAG PIPELINE WITH DEGRADED FALLBACK ---
class ResilientDegradedRAG:
    def __init__(self, tpm_threshold: int = 12000):
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_collection(name="research_papers")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.cache = ExactResponseCache()
        
        self.tpm_threshold = tpm_threshold
        self.current_minute_tokens = 0
        self.last_reset_time = time.time()

    def _update_token_counter(self, tokens_used: int):
        now = time.time()
        if now - self.last_reset_time > 60:
            self.current_minute_tokens = 0
            self.last_reset_time = now
        self.current_minute_tokens += tokens_used

    # def is_degraded_mode() -> bool:
    #     return self.current_minute_tokens > (0.8 * self.tpm_threshold)
    # Fixed: Added 'self' parameter inside parentheses
    def is_degraded_mode(self) -> bool:
        return self.current_minute_tokens > (0.8 * self.tpm_threshold)
    async def run_query(self, query: str) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        # Determine Top-K Context Based on System Load (Degraded Mode check)
        degraded = self.is_degraded_mode()
        top_k = 2 if degraded else 4
        
        # Vector Retrieval Tool Call
        query_emb = self.embedder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_emb, n_results=top_k)
        
        docs = results["documents"][0] if results and results["documents"] else []
        metas = results["metadatas"][0] if results and results["metadatas"] else []
        
        chunks = [{"text": d, "meta": m} for d, m in zip(docs, metas)]
        chunk_texts = [c["text"] for c in chunks]

        # Check Cache
        cached_val = self.cache.get(query, chunk_texts)
        if cached_val:
            return {
                "query": query,
                "answer": cached_val,
                "source": "exact_cache",
                "total_ms": (time.perf_counter() - start_time) * 1000,
                "tokens_used": 0,
                "degraded_mode": degraded
            }

        # Simulated dynamic token calculation
        estimated_tokens = sum(len(c["text"].split()) for c in chunks) + 50
        self._update_token_counter(estimated_tokens)

        # Formatted Response with strict citation
        simulated_response = (
            f"According to [{metas[0]['paper']}, {metas[0]['section']}, Page {metas[0]['page']}], "
            f"the core mechanism leverages key transformer sequence operations."
        )
        
        if degraded:
            simulated_response += " [Note: Answer generated under Rate-Limit Degraded Context Mode]"

        self.cache.set(query, chunk_texts, simulated_response)

        return {
            "query": query,
            "answer": simulated_response,
            "source": "llm_generation",
            "total_ms": (time.perf_counter() - start_time) * 1000,
            "tokens_used": estimated_tokens,
            "degraded_mode": degraded
        }

if __name__ == "__main__":
    async def demo():
        pipeline = ResilientDegradedRAG()
        res = await pipeline.run_query("What is multi-head attention?")
        print("Pipeline Execution Output:")
        print(f"- Answer: {res['answer']}")
        print(f"- Latency: {res['total_ms']:.2f} ms")
        print(f"- Degraded Mode Active: {res['degraded_mode']}")
        
    asyncio.run(demo())