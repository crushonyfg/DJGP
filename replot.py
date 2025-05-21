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
    horizontal lines showing each method's mean RMSE across runs.
    """
    metrics = ['rmse', 'crps']
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes = axes.ravel()

    # Color and marker map for each base method
    colors = {
        'JumpGP':    'blue',
        'JumpGPsir': 'red',
        'DeepGP':    'green',
        'LMJGP':      'cyan',
        'GPsir':     'purple',
        'GP':        'orange',
        'NNJGP':     'magenta',
        'BNNJGP':    'brown',
    }
    markers = {
        'JumpGP':    'o',
        'JumpGPsir': 's',
        'DeepGP':    '^',
        'LMJGP':      '*',
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


import matplotlib.pyplot as plt
import numpy as np
import torch

# def plot_metrics_boxplot(results: dict, output_file: str):
#     """
#     对每个方法在 RMSE 和 CRPS 上绘制 boxplot，不再画散点和均值直线。
#     results: {
#         run_id: {
#             method_name: (rmse, crps, runtime),
#             ...
#         },
#         ...
#     }
#     output_file: 保存的文件路径
#     """
#     metrics = ['rmse', 'crps']
#     fig, axes = plt.subplots(1, 2, figsize=(15, 7))
#     axes = axes.ravel()

#     # 方法对应的颜色
#     colors = {
#         'JumpGP':    'blue',
#         'JumpGPsirGlobal': 'red',
#         'DeepGP':    'green',
#         'LMJGP':      'cyan',
#         'GPsir':     'purple',
#         'GP':        'orange',
#         'NNJGP':     'magenta',
#         'BNNJGP':    'brown',
#     }

#     # 收集每个方法在各 runs 上的两个指标
#     metric_by_method = {base: [] for base in colors.keys()}
#     for run_res in results.values():
#         for method, vals in run_res.items():
#             base = method.split('_')[0]
#             # 提取 rmse, crps
#             raw_rmse, raw_crps = vals[0], vals[1]
#             rmse = (raw_rmse.detach().cpu().item()
#                     if isinstance(raw_rmse, torch.Tensor)
#                     else float(raw_rmse))
#             crps = (raw_crps.detach().cpu().item()
#                     if isinstance(raw_crps, torch.Tensor)
#                     else float(raw_crps))
#             metric_by_method[base].append((rmse, crps))

#     # 对每个指标画 boxplot
#     for i, metric in enumerate(metrics):
#         ax = axes[i]
#         # 准备数据：一个 list of lists
#         labels = list(metric_by_method.keys())
#         data = []
#         for base in labels:
#             # 对应每个 run 的第 i 个指标
#             vals = [entry[i] for entry in metric_by_method[base]]
#             data.append(vals if vals else [np.nan])  # 若某方法无数据，放一个 nan 保持位置

#         # 绘制 boxplot，使用 patch_artist 方便上色
#         bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
#         for patch, base in zip(bp['boxes'], labels):
#             patch.set_facecolor(colors.get(base, 'white'))
#             patch.set_edgecolor('black')

#         ax.set_title(f'{metric.upper()} by Method')
#         ax.set_ylabel(metric.upper())
#         ax.grid(True, axis='y')
#         ax.tick_params(axis='x', rotation=45)

#     plt.tight_layout()
#     plt.savefig(output_file, bbox_inches='tight', dpi=300)
#     plt.close()
#     print(f"Saved boxplot to {output_file}")

def plot_metrics_boxplot(results: dict, output_file: str):
    """
    对每个方法在 RMSE 和 CRPS 上绘制 boxplot，方法动态识别，
    不再假定事先知道有哪些 base。
    """
    metrics = ['rmse', 'crps']
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes = axes.ravel()

    # 预定义常见方法的颜色，未知方法使用 'gray'
    colors = {
        'JumpGP':    'blue',
        'JumpGPsirGlobal': 'red',
        'DeepGP':    'green',
        'LMJGP':      'cyan',
        'GPsir':     'purple',
        'GP':        'orange',
        'NNJGP':     'magenta',
        'BNNJGP':    'brown',
    }
    default_color = 'gray'

    # 动态收集每个 base 方法在所有 runs 上的 (rmse, crps)
    metric_by_method = {}
    for run_res in results.values():
        for method, vals in run_res.items():
            base = method.split('_')[0]
            # 将 DJGP 替换为 LMJGP
            if base == 'DJGP':
                base = 'LMJGP'
            # 提取指标值
            raw_rmse, raw_crps = vals[0], vals[1]
            rmse = (raw_rmse.detach().cpu().item()
                    if isinstance(raw_rmse, torch.Tensor)
                    else float(raw_rmse))
            crps = (raw_crps.detach().cpu().item()
                    if isinstance(raw_crps, torch.Tensor)
                    else float(raw_crps))
            metric_by_method.setdefault(base, []).append((rmse, crps))

    # 准备排序后的标签列表，保证图中方法顺序一致
    labels = sorted(metric_by_method.keys())

    # 对每个指标画 boxplot
    for i, metric in enumerate(metrics):
        ax = axes[i]
        data = [
            [entry[i] for entry in metric_by_method[base]]
            for base in labels
        ]

        # 绘制 boxplot
        bp = ax.boxplot(
            data,
            tick_labels=labels,
            patch_artist=True,
            showfliers=False
        )
        # 根据方法名上色
        for patch, base in zip(bp['boxes'], labels):
            color = colors.get(base, default_color)
            # patch.set_facecolor(color)
            patch.set_facecolor('gray')
            patch.set_edgecolor('black')

        ax.set_title(f'{metric.upper()} by Method')
        ax.set_ylabel(metric.upper())
        ax.grid(True, axis='y')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved boxplot to {output_file}")


import os
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
    parser.add_argument(
        '--output-file-boxplot',
        type=str,
        default='metrics_with_boxplot.png',
        help='Filename for the output plot'
    )
    parser.add_argument(
        '--boxplot',
        type=bool,
        default=True,
        help='Whether to plot boxplot'
    )
    args = parser.parse_args()

    results = load_results(args.results_pkl)
    # plot_metrics_with_mean(results, args.output_file)
    if args.boxplot:
        parent_dir = os.path.dirname(args.results_pkl)
        output_file_boxplot = os.path.join(parent_dir, args.output_file_boxplot)
        plot_metrics_boxplot(results, output_file_boxplot)
    else:
        plot_metrics_with_mean(results, args.output_file)

if __name__ == "__main__":
    main()
