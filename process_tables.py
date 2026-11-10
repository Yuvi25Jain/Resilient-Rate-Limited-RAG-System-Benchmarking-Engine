import csv
import numpy as np

def compute_percentiles(values):
    if not values:
        return 0.0, 0.0, 0.0
    return (
        np.percentile(values, 50),
        np.percentile(values, 95),
        np.percentile(values, 99)
    )

def analyze_results():
    scenarios = {}
    
    with open("results.csv", mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sc = row["scenario"]
            if sc not in scenarios:
                scenarios[sc] = {
                    "source": row["source"],
                    "first_tokens": [],
                    "totals": [],
                    "statuses": [],
                    "tokens_out": [],
                    "retries": []
                }
            
            scenarios[sc]["first_tokens"].append(float(row["first_token_ms"]))
            scenarios[sc]["totals"].append(float(row["total_ms"]))
            scenarios[sc]["statuses"].append(row["status"])
            scenarios[sc]["tokens_out"].append(int(row["tokens_out"]))
            scenarios[sc]["retries"].append(int(row["retries"]))

    print("\n========================================================")
    print("TABLE 2a: Latency Metrics (p50 / p95 / p99 in ms)")
    print("========================================================")
    print(f"{'Scenario':<20} | {'Source':<10} | {'TTFT p50/p95/p99 (ms)':<25} | {'Total p50/p95/p99 (ms)':<25}")
    print("-" * 88)

    for sc, data in scenarios.items():
        ft_50, ft_95, ft_99 = compute_percentiles(data["first_tokens"])
        tot_50, tot_95, tot_99 = compute_percentiles(data["totals"])
        
        ft_str = f"{ft_50:.1f} / {ft_95:.1f} / {ft_99:.1f}"
        tot_str = f"{tot_50:.1f} / {tot_95:.1f} / {tot_99:.1f}"
        
        print(f"{sc:<20} | {data['source']:<10} | {ft_str:<25} | {tot_str:<25}")

    print("\n========================================================")
    print("TABLE 2b: Outcome Metrics")
    print("========================================================")
    print(f"{'Scenario':<20} | {'Succeeded %':<12} | {'Rate-Limited %':<15} | {'Avg Tokens/Ans':<15}")
    print("-" * 70)

    for sc, data in scenarios.items():
        total_reqs = len(data["statuses"])
        succ = (data["statuses"].count("ok") / total_reqs) * 100
        rl = (data["statuses"].count("rate_limited") / total_reqs) * 100
        avg_tok = np.mean(data["tokens_out"]) if data["tokens_out"] else 0
        
        print(f"{sc:<20} | {succ:<12.1f} | {rl:<15.1f} | {avg_tok:<15.1f}")

if __name__ == "__main__":
    analyze_results()