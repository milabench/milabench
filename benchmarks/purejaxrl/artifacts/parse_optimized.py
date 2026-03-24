import json
import statistics

run_files = ['/home/delaunap/milabench/benchmarks/purejaxrl/artifacts/benchmarks/optimized/run_1.log', 
             '/home/delaunap/milabench/benchmarks/purejaxrl/artifacts/benchmarks/optimized/run_2.log', 
             '/home/delaunap/milabench/benchmarks/purejaxrl/artifacts/benchmarks/optimized/run_3.log']

rates = []
rewards = []

for f in run_files:
    data = []
    with open(f, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('{'):
                try:
                    data.append(json.loads(line))
                except:
                    pass
    
    train_rates = [d['rate'] for d in data if d.get('task') == 'train' and 'rate' in d]
    train_returns = [d['returns'] for d in data if d.get('task') == 'train' and 'returns' in d]
    
    if train_rates and train_returns:
        rates.append(train_rates[-1])
        rewards.append(max(train_returns))

median_rate = round(statistics.median(rates), 1)
best_reward = round(max(rewards), 1)

with open('/home/delaunap/milabench/benchmarks/purejaxrl/artifacts/benchmarks/results.csv', 'a') as csv:
    csv.write(f"optimized,{median_rate},{best_reward}\n")
    
print(f"Appended optimized,{median_rate},{best_reward} to results.csv")
