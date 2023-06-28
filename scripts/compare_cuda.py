# Script to compare performance of CUDA implementations.

from collections import defaultdict
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')

results_dir = pathlib.Path("results/cuda/")
results_dir.mkdir(parents=True, exist_ok=True)
stats_file = open(results_dir / "README.md", "w")

for csv_name in ["iter.log", "main.log"]:
    print(f"# {csv_name}", file=stats_file)

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    block_sizes = set()
    for log_path in pathlib.Path("runs/cuda").rglob(csv_name):
        # e.g. ./runs/cuda/cuda-1/800/1/iter.log
        name = log_path.parts[-4]
        grid_size = int(log_path.parts[-3])
        df = pd.read_csv(log_path)
        for k, v in df[df > 0].mean().items():
            results[k][name][grid_size].append(v)

    title_map = { 
        "update": "Iteration time", 
        "image_draw": "Time to draw image", 
        "total": "Total time" 
    }
    for k, d in results.items():
        print(f"## {k}", file=stats_file)
        df = pd.DataFrame(d)
        df = df.applymap(np.mean)  # Calculate mean for each list
        df = df.sort_index()
        df = df.sort_index(axis=1)
        print(df.to_markdown(floatfmt=".4f"), file=stats_file)
        print("\n\n", file=stats_file)

        speedup_df = pd.DataFrame()
        for name in ["cuda-2", "cuda-3"]:
            speedup_df[f"s_{name}"] = df["cuda-1"] / df[name]
        print(speedup_df.to_markdown(floatfmt=".4f"), file=stats_file)

        if k in title_map:
            fig, ax = plt.subplots()
            df.plot(title=title_map[k], ylabel="Time [s]", xlabel="Grid size [N x N]", ax=ax)
            fig.savefig(f"{results_dir}/{csv_name.split('.')[0]}-{k}.png", bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots()
            speedup_df.filter(like="s_").plot(title=title_map[k] + " speedup", ylabel="Speedup (vs 1 GPU basic)", xlabel="Grid size [N x N]", ax=ax)
            fig.savefig(f"{results_dir}/speedup-{csv_name.split('.')[0]}-{k}.png", bbox_inches="tight")
            plt.close(fig)
