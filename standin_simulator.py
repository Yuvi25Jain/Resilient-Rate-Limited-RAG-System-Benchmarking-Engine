import asyncio
import random
import time
from typing import AsyncGenerator, Dict, Any

class StandInModelService:
    """
    Stand-in Mock Model Service calibrated from real Groq measurements.
    Simulates:
    - Base latency (TTFT & TPOT) with normal distribution jitter
    - Concurrency queue slowdown under heavy load
    - Token Bucket Rate Limiting (HTTP 429 Refusal)
    - Token-by-token async streaming output
    """
    def __init__(self, rpm_limit: int = 30, tpm_limit: int = 14400):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.in_flight_requests = 0
        self.request_timestamps = []
        
        # Calibrated latencies based on measured real Groq API run (~467 ms base)
        self.base_ttft_p50 = 0.450  # 450 ms base median TTFT
        self.base_tpot = 0.012      # 12 ms per output token

    async def generate_stream(self, prompt: str, max_tokens: int = 120) -> AsyncGenerator[Dict[str, Any], None]:
        current_time = time.time()
        # Clean timestamps older than 60 seconds
        self.request_timestamps = [t for t in self.request_timestamps if current_time - t < 60]

        # 1. Check Rate Limit (RPM Threshold)
        if len(self.request_timestamps) >= self.rpm_limit:
            yield {
                "type": "error",
                "status": "rate_limited",
                "error_code": 429,
                "message": "Rate limit reached: HTTP 429 Too Many Requests (Stand-in Simulator)"
            }
            return

        self.request_timestamps.append(current_time)
        self.in_flight_requests += 1

        try:
            # 2. Simulate Latency Jitter + Queue Penalty under Load
            concurrency_penalty = max(0, (self.in_flight_requests - 1) * 0.04)
            ttft = self.base_ttft_p50 + random.uniform(0.01, 0.05) + concurrency_penalty
            
            await asyncio.sleep(ttft)
            yield {"type": "first_token", "ttft_ms": ttft * 1000}

            # 3. Output Streamed Response Chunks
            simulated_text = (
                "Based on [Attention Is All You Need, Section 3.2, Page 4], "
                "Multi-Head Attention allows the model to jointly attend to information from different subspaces."
            )
            words = simulated_text.split()
            
            for word in words:
                await asyncio.sleep(self.base_tpot)
                yield {"type": "token", "delta": word + " "}

            yield {"type": "done", "status": "ok"}
            
        finally:
            self.in_flight_requests -= 1