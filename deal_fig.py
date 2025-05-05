import os
import shutil
import re

# 原始目录和目标目录
root_dir = 'D:/new_windows/PhD/spring2025/park/highJGP/code/exps'
target_dir = os.path.join(root_dir, 'figs')

# 创建目标目录（如果不存在）
os.makedirs(target_dir, exist_ok=True)

# 遍历E:/下所有文件夹
for item in os.listdir(root_dir):
    match = re.match(r'exp(\d+)', item)
    if match:
        i = match.group(1)
        source_path = os.path.join(root_dir, item, 'metrics_comparison.png')
        if os.path.isfile(source_path):
            target_path = os.path.join(target_dir, f'metrics_comparison{i}.png')
            shutil.copyfile(source_path, target_path)
            print(f'复制: {source_path} -> {target_path}')
        else:
            print(f'未找到文件: {source_path}')
