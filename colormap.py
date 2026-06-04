import numpy as np
import matplotlib.pyplot as plt

for name in ['inferno', 'viridis']:
    cmap = plt.get_cmap(name)
    print(f"\n// {name}")
    for i in range(256):
        r,g,b,_ = [int(x*255) for x in cmap(i/255)]
        print(f"    {{{b},{g},{r}}},")  # BGR order