import argparse
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_results(filename: str) -> dict:
    """Load the results dict from a pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def plot_metrics_with_mean(results: dict, output_file: str):
    """
    Plot RMSE and CRPS vs runtime, and on the RMSE subplot draw
    horizontal lines showing each method’s mean RMSE across runs.
    """
    metrics = ['rmse', 'crps']
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes = axes.ravel()

    # Color and marker map for each base method
    colors = {
        'JumpGP':    'blue',
        'JumpGPsir': 'red',
        'DeepGP':    'green',
        'DJGP':      'cyan',
        'GPsir':     'purple',
        'GP':        'orange',
        'NNJGP':     'magenta',
        'BNNJGP':    'brown',
    }
    markers = {
        'JumpGP':    'o',
        'JumpGPsir': 's',
        'DeepGP':    '^',
        'DJGP':      '*',
        'GPsir':     'D',
        'GP':        'v',
        'NNJGP':     'X',
        'BNNJGP':    'P',
    }

    # 1) Compute mean RMSE per base method
    rmse_by_method = {}
    for run_res in results.values():
        for method, vals in run_res.items():
            base = method.split('_')[0]
            raw_rmse = vals[0]
            rmse = (raw_rmse.detach().cpu().item()
                    if isinstance(raw_rmse, torch.Tensor)
                    else float(raw_rmse))
            rmse_by_method.setdefault(base, []).append(rmse)
    mean_rmse = {base: np.mean(vals) for base, vals in rmse_by_method.items()}

    # 2) Plot scatter for each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for run_idx, run_res in enumerate(results.values(), start=1):
            for method, vals in run_res.items():
                base = method.split('_')[0]
                raw_time = vals[2]  # runtime is at index 2
                raw_score = vals[i] # rmse at 0, crps at 1
                x = (raw_time.detach().cpu().item()
                     if isinstance(raw_time, torch.Tensor)
                     else float(raw_time))
                y = (raw_score.detach().cpu().item()
                     if isinstance(raw_score, torch.Tensor)
                     else float(raw_score))
                ax.scatter(
                    x, y,
                    color=colors.get(base, 'black'),
                    marker=markers.get(base, 'x'),
                    label=base
                )
                ax.annotate(
                    str(run_idx),
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8
                )

        ax.set_xlabel('Runtime (s)')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} vs Runtime')
        ax.grid(True)

        # Deduplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(
            unique.values(),
            unique.keys(),
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )

        # 3) On RMSE subplot, draw horizontal mean lines
        if metric == 'rmse':
            for base, m in mean_rmse.items():
                ax.axhline(
                    y=m,
                    color=colors.get(base, 'black'),
                    linestyle='-',
                    linewidth=1.5,
                    label=f'{base} mean'
                )
            # Update legend to include mean lines
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(
                unique.values(),
                unique.keys(),
                bbox_to_anchor=(1.05, 1),
                loc='upper left'
            )

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved plot with mean lines to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Load a results.pkl and plot RMSE/CRPS vs runtime "
                    "with average RMSE lines."
    )
    parser.add_argument(
        '--results-pkl',
        type=str,
        required=True,
        help='Path to the pickle file containing the results dict'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='metrics_with_means.png',
        help='Filename for the output plot'
    )
    args = parser.parse_args()

    results = load_results(args.results_pkl)
    plot_metrics_with_mean(results, args.output_file)

if __name__ == "__main__":
    main()
