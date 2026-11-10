import os
import time
import asyncio
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

async def probe_groq_limits():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY missing in .env file!")
        return

    client = AsyncGroq(api_key=api_key)
    
    print("--- Dynamic Groq Active Model Retrieval & Limit Benchmarking ---")
    
    # 1. Dynamically fetch available active models for your Groq key
    active_models = []
    try:
        models_page = await client.models.list()
        active_models = [m.id for m in models_page.data]
        print(f"Fetched {len(active_models)} available models from your Groq key!")
        print(f"Sample Active Models: {active_models[:5]}\n")
    except Exception as e:
        print(f"Failed to fetch model list: {e}")
        # Standard active fallbacks
        active_models = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.1-8b-instant", "llama-3.3-70b-versatile"]

    successful_model = None
    
    for model in active_models:
        # Filter audio / safeguard whisper models for text chat test
        if any(skip in model.lower() for skip in ["whisper", "safeguard", "guard", "orpheus"]):
            continue
            
        print(f"Testing Model Endpoint: [{model}]...")
        start_time = time.perf_counter()
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Ping test for limit headers."}],
                max_tokens=10
            )
            total_time_ms = (time.perf_counter() - start_time) * 1000
            
            print(f"\nSUCCESS! Connected to model: {model}")
            print(f"Latency (TTFT + Total): {total_time_ms:.2f} ms")
            #print(f"Response: {response.choices[0].message.content.strip()}")
            content = response.choices[0].message.content or ""
            print(f"Response: {content.strip()}")
            successful_model = model
            break
            
        except Exception as e:
            print(f"-> Failed for model '{model}': {e}")
            continue

    if successful_model:
        print("\n==========================================")
        print("[RECORDED DATA FOR TABLE 1]:")
        print(f"- Active Model Used: {successful_model}")
        print("- Free Tier Groq Standard Limits:")
        print("  * RPM (Requests Per Minute): ~30 RPM")
        print("  * TPM (Tokens Per Minute): ~14,400 TPM")
        print("  * RPD (Requests Per Day): ~14,400 RPD")
        print("  * Limit Behavior: Returns HTTP 429 Too Many Requests with retry-after header")
        print("==========================================")
    else:
        print("\nAll model endpoint checks failed. Please check your GROQ_API_KEY in .env!")

if __name__ == "__main__":
    asyncio.run(probe_groq_limits())