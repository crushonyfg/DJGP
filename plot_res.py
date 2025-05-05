# python plot_res.py --input results.pkl --output metrics_comparison.png
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pickle
import matplotlib.pyplot as plt
import argparse

def load_results(filename='results.pkl'):
    """从文件加载结果"""
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    print(f"Results loaded from {filename}")
    return results

def plot_metrics(results, output_filename='metrics_comparison1.png'):
    """绘制散点图"""
    metrics = ['rmse', 'q25', 'q50', 'q75']
    metrics_names = ['RMSE', 'NLPD 25th Percentile', 'NLPD 50th Percentile', 'NLPD 75th Percentile']
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    # 为所有方法定义颜色和标记
    colors = {
        'JumpGP': 'blue',
        'JumpGP_sir': 'red',
        'DeepGP': 'green',
        'GP_sir': 'purple',
        'GP': 'orange'
    }
    
    markers = {
        'JumpGP': 'o',
        'JumpGP_sir': 's',
        'DeepGP': '^',
        'GP_sir': 'D',
        'GP': 'v'
    }
    
    # 用于添加到图例中的方法名与标记的映射
    legend_elements = {}
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metrics_names)):
        ax = axes[idx]
        
        for folder_name, res_i in results.items():
            for method, values in res_i.items():
                # 安全地提取基础方法名
                if '_' in method:
                    # 检查方法名是否包含'sir'
                    if 'sir' in method.lower():
                        # 如果方法名像'GP_sir_0'，提取为'GP_sir'
                        parts = method.split('_')
                        if len(parts) > 2:
                            base_method = f"{parts[0]}_{parts[1]}"
                        else:
                            base_method = method
                    else:
                        # 如果方法名像'DeepGP_0'，提取为'DeepGP'
                        base_method = method.split('_')[0]
                else:
                    base_method = method
                
                # 确保base_method在colors字典中
                if base_method not in colors:
                    print(f"Warning: Method '{base_method}' not found in colors dictionary. Using default color.")
                    colors[base_method] = 'gray'
                    markers[base_method] = '*'
                
                # 绘制散点图
                scatter = ax.scatter(values[4], values[idx], 
                          color=colors[base_method],
                          marker=markers[base_method],
                          alpha=0.7)
                
                # 为图例准备
                if base_method not in legend_elements:
                    legend_elements[base_method] = (scatter, base_method)
        
        ax.set_xlabel('Runtime (s)', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} vs Runtime', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 设置坐标轴刻度字体大小
        ax.tick_params(axis='both', which='major', labelsize=10)
        
    # 添加自定义图例
    legend_handles = [handle for handle, _ in legend_elements.values()]
    legend_labels = [label for _, label in legend_elements.values()]
    
    # 将图例放在图的下方
    fig.legend(legend_handles, legend_labels, 
               loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), 
               ncol=len(legend_elements), 
               fontsize=12)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 为底部图例留出空间
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Plot saved as {output_filename}")

def main():
    parser = argparse.ArgumentParser(description='Plot results from pickle file')
    parser.add_argument('--input', type=str, default='results.pkl', help='Input pickle file containing results')
    parser.add_argument('--output', type=str, default='metrics_comparison.png', help='Output image file')
    args = parser.parse_args()
    
    # 加载结果
    results = load_results(args.input)
    
    # 打印结果的结构
    print("\nResults structure:")
    for folder_name, res_i in results.items():
        print(f"Folder: {folder_name}")
        for method, values in res_i.items():
            print(f"  Method: {method}, Values shape: {len(values)}")
            if len(values) >= 5:
                print(f"    RMSE: {values[0]:.4f}, q25: {values[1]:.4f}, q50: {values[2]:.4f}, q75: {values[3]:.4f}, Runtime: {values[4]:.2f}s")
    
    # 绘制图表
    plot_metrics(results, args.output)

if __name__ == "__main__":
    main()
