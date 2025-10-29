import pandas as pd

# Load all CSVs
files = ["output/array_cpu/comp_pi_39549768.csv", "output/array_cpu/comp_pi_39550471.csv", "output/array_cpu/comp_pi_39550491.csv"]
dfs = [pd.read_csv(f) for f in files]

# Ensure alignment by 'num_threads'
merged = pd.concat(dfs).groupby("num_threads", as_index=False).mean()

# Save averaged data
merged.to_csv("output/array_cpu/comp_pi_avg.csv", index=False)
print("Saved averaged results to comp_pi_avg.csv")
