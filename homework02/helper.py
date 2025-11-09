import pandas as pd
import glob
import math

# # Load all CSVs
files = glob.glob("output/300M/prefix*local.csv")
dfs = [pd.read_csv(f) for f in files]

# # # Ensure alignment by 'num_threads'
merged = pd.concat(dfs).groupby("num_threads", as_index=False).mean()
merged.to_csv("output/300M/prefix_avg_local_300M.csv", index=False)

# # Save averaged data
print(merged)

# Get columns related to pi calculation
# pi_columns = [col for col in merged.columns if col.startswith('pi_')]

# Calculate the average value for each pi column
# pi_averages = merged[pi_columns].mean()

# # Print the averages
# print("Averages for pi calculations per method:")
# for col, avg in pi_averages.items():
#     print(f"{col}: {avg}")
#     print(f"Error: {(abs(avg - math.pi) / ((avg +math.pi)/2)) * 100}%")
