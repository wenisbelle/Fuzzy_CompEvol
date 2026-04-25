import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import os


def parse_ga_logbook(filepath: str):
    """
    Parses a GA logbook .txt file and extracts generation stats.
    Returns a dict with lists: gen, nevals, avg, std, min, max.
    """
    data = {col: [] for col in ["gen", "nevals", "avg", "std", "min", "max"]}

    with open(filepath, "r") as f:
        lines = f.readlines()

    # Find the header line
    header_idx = None
    for i, line in enumerate(lines):
        if re.match(r"\s*gen\s+nevals", line):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Could not find the 'gen nevals avg std min max' header in: {filepath}")

    # Parse data rows after header
    for line in lines[header_idx + 1:]:
        line = line.strip()
        if not line or line.startswith("==="):
            break
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            data["gen"].append(int(parts[0]))
            data["nevals"].append(int(parts[1]))
            data["avg"].append(float(parts[2]))
            data["std"].append(float(parts[3]))
            data["min"].append(float(parts[4]))
            data["max"].append(float(parts[5]))
        except ValueError:
            continue

    return data


def filter_up_to_generation(data: dict, max_gen: int):
    """Filters all data lists to only include entries up to max_gen (inclusive)."""
    cutoff = next(
        (i for i, g in enumerate(data["gen"]) if g > max_gen),
        len(data["gen"])
    )
    return {col: data[col][:cutoff] for col in data}


def plot_multi_convergence(logbooks: list, labels: list, max_gen: int = 20, output_path: str = None):
    """
    Plots the minimum fitness across generations for multiple logbook files
    on the same axes, up to max_gen.
    """
    colors = ["#2563EB", "#16A34A", "#DC2626", "#D97706"]
    markers = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, (data_raw, label) in enumerate(zip(logbooks, labels)):
        data = filter_up_to_generation(data_raw, max_gen)
        generations = data["gen"]
        min_values  = data["min"]
        color       = colors[i % len(colors)]
        marker      = markers[i % len(markers)]

        ax.plot(generations, min_values,
                color=color, linewidth=2.0, label=label, zorder=3)

        # Mark best point for each run
        best_idx = min_values.index(min(min_values))
        ax.scatter(generations[best_idx], min_values[best_idx],
                   color=color, marker=marker, s=70, zorder=5,
                   edgecolors="white", linewidths=0.8)

        # Light shaded band under each curve
        ax.fill_between(generations, min_values, alpha=0.05, color=color)

        print(f"[{label}] Best min fitness up to gen {max_gen}: "
              f"{min(min_values):,.4f} at generation {generations[best_idx]}")

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Minimum Fitness (Cost)", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(left=0, right=max_gen)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot min fitness convergence for exactly 4 GA logbook files up to a given generation."
    )
    parser.add_argument(
        "logbooks",
        type=str,
        nargs=4,
        help="Paths to 4 GA logbook .txt files, e.g.: log1.txt log2.txt log3.txt log4.txt"
    )
    parser.add_argument(
        "--labels", "-l",
        type=str,
        nargs=4,
        default=None,
        help="Labels for each logbook in the legend (4 values). "
             "Defaults to the filenames if not provided."
    )
    parser.add_argument(
        "--max-gen", "-g",
        type=int,
        default=20,
        help="Maximum generation to plot up to (default: 30)."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Optional output path to save the plot (e.g. convergence_multi.png). "
             "If not provided, the plot is displayed interactively."
    )
    args = parser.parse_args()

    # Validate files
    for path in args.logbooks:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

    # Build labels
    labels = args.labels if args.labels else [os.path.basename(p) for p in args.logbooks]

    # Parse all logbooks
    logbooks = []
    for path, label in zip(args.logbooks, labels):
        data = parse_ga_logbook(path)
        logbooks.append(data)
        print(f"Parsed '{label}': {len(data['gen'])} generations total.")

    print()
    plot_multi_convergence(logbooks, labels, max_gen=args.max_gen, output_path=args.output)


if __name__ == "__main__":
    main()