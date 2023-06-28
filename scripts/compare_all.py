# Script to compare performance of all implementations.

from collections import defaultdict
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')

results_dir = pathlib.Path("results/all/")
results_dir.mkdir(parents=True, exist_ok=True)
stats_file = open(results_dir / "README.md", "w")

for csv_name in ["iter.log", "main.log"]:
    print(f"# {csv_name}", file=stats_file)

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    results2 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for log_path in pathlib.Path("runs/all").rglob(csv_name):    
        # e.g. /runs/all/cuda/400/1/iter.log
        name = log_path.parts[-4]
        grid_size = int(log_path.parts[-3])
        df = pd.read_csv(log_path)
        for k, v in df[df > 0].mean().items():
            results[k][name][grid_size].append(v)
            results2[grid_size][k][name].append(v)

    special_size = 3200
    title_map = { 
        "update": "Iteration time", 
        "image_draw": "Time to draw image", 
        "total": "Total time" 
    }
    names = ["serial", "openmp", "cuda-1", "cuda-2", "cuda-3"]
    for k, d in results.items():
        print(f"## {k}", file=stats_file)
        df = pd.DataFrame(d)
        df = df.applymap(np.mean)  # Calculate mean for each list
        df = df.sort_index()
        df = df.reindex(names, axis=1)
        # df = df.sort_index(axis=1)
        print(df.to_markdown(floatfmt=".4f"), file=stats_file)
        print("\n\n", file=stats_file)

        OPENMP_P = 32
        speedup_df = pd.DataFrame()
        speedup_df["s_openmp"] = df["serial"] / df["openmp"]
        speedup_df["e_openmp"] = speedup_df["s_openmp"] / OPENMP_P
        speedup_df["s_cuda-1"] = df["serial"] / df["cuda-1"]
        speedup_df["s_cuda-2"] = df["serial"] / df["cuda-2"]
        speedup_df["s_cuda-3"] = df["serial"] / df["cuda-3"]
        print(speedup_df.to_markdown(floatfmt=".4f"), file=stats_file)

        if k in title_map:
            fig, ax = plt.subplots()
            df.plot(title=title_map[k], ylabel="Time [s]", xlabel="Grid size [N x N]", ax=ax, logy=True)
            fig.savefig(f"{results_dir}/{csv_name.split('.')[0]}-{k}.png", bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots()
            speedup_df.filter(like="s_").plot(title=title_map[k] + " speedup", ylabel="Speedup (vs serial)", xlabel="Grid size [N x N]", ax=ax)
            fig.savefig(f"{results_dir}/speedup-{csv_name.split('.')[0]}-{k}.png", bbox_inches="tight")
            plt.close(fig)

    df = pd.DataFrame(results2[special_size])
    df = df.applymap(np.mean)  # Calculate mean for each list
    # df = df.sort_index()
    df = df.reindex(names, axis=0)
    fig, ax = plt.subplots()
    title = "Iteration time" if csv_name == "iter.log" else "Total time"
    df = df.drop("image_write", axis=1) if "image_write" in df.columns else df
    df.plot.bar(title=f"{title}", ylabel="Time [s]", ax=ax, logy=True)
    fig.savefig(f"{results_dir}/{csv_name.split('.')[0]}-overall.png", bbox_inches="tight")
    plt.close(fig)
