import numpy as np

values = np.loadtxt("outputs/center_x_log.txt")

def print_stats(data, label=""):
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)
    #p5 = np.percentile(data, 5)
    #p95 = np.percentile(data, 95)

    print(f"\n--- {label} ---")
    print(f"N: {len(data)}")
    print(f"Media: {mean:.2f}")
    print(f"Mediana: {median:.2f}")
    print(f"Std: {std:.2f}")
    print(f"Min: {min_val}")
    print(f"Max: {max_val}")
    print(f"Rango: {max_val - min_val}")
    #print(f"P5: {p5}")
    #print(f"P95: {p95}")
    #print(f"Rango 90%: {p95 - p5:.2f}")

print_stats(values, label="GLOBAL")

block_size = 300
num_blocks = len(values) // block_size

for i in range(num_blocks):
    start = i * block_size
    end = start + block_size
    block = values[start:end]

    print_stats(block, label=f"BLOQUE {i+1} ({start}-{end})")