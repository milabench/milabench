import json
import statistics

def process_file(filepath):
    current_run = "WARMUP"
    runs = {"WARMUP": [], "RUN 1": [], "RUN 2": [], "RUN 3": []}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line in runs:
                current_run = line
                continue
            if line.startswith('{') and current_run:
                try:
                    runs[current_run].append(json.loads(line))
                except json.JSONDecodeError:
                    pass
                
    rates = []
    rewards = []
    
    for run_name, data in runs.items():
        if run_name == "WARMUP":
            continue
        if not data: continue
        
        train_rates = [d['rate'] for d in data if d.get('task') == 'train' and 'rate' in d]
        train_returns = [d['returns'] for d in data if d.get('task') == 'train' and 'returns' in d]
        
        # Often the final rate reported is the overall throughput, or we can take the median.
        # Let's take the very last reported rate for the run as the cumulative steps/second.
        # Wait, if `benchmate.timings.StepTimer` is used, the final rate logged is the rate over the whole execution if `delta` accumulates, or just the last 1000 steps. 
        # Actually, let's print the last rate, median of all instantaneous rates, and best reward for context.
        final_rate = train_rates[-1] if train_rates else 0
        median_rate = statistics.median(train_rates) if train_rates else 0
        best_reward = max(train_returns) if train_returns else 0
        
        print(f"--- {run_name} ---")
        print(f"Final rate: {final_rate}")
        print(f"Median rate: {median_rate}")
        print(f"Best mean reward: {best_reward}")
        
        rates.append(final_rate)
        rewards.append(best_reward)
        
    print("\n--- SUMMARY ---")
    print(f"Final rates per run: {rates}")
    if rates:
        print(f"Median of final rates: {statistics.median(rates)}")
    print(f"Best rewards per run: {rewards}")
    if rewards:
        print(f"Best reward overall: {max(rewards)}")

if __name__ == '__main__':
    process_file('/home/delaunap/milabench/benchmarks/purejaxrl/artifacts/benchmarks/baseline.txt')
