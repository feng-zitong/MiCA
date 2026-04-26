#!/bin/bash

# 切换到项目根目录
cd "$(dirname "$0")/.."

# ======================== 训练配置 ========================
dataset_name="refcoco" # "refcoco", "refcoco+", "refcocog_g", "refcocog_u"
config_name="refcoco/MiCA_base.yaml"

# GPU配置 - 修改这里来选择您要使用的GPU
gpu="5,6,7"  # 指定要使用的GPU编号，用逗号分隔

# 其他训练参数
omp=8          # CPU线程数
master_port=29597 # 统一使用一个端口

# ======================== 自动计算进程数 ========================
np=$(echo $gpu | tr -cd ',' | wc -c)
np=$((np + 1))

# ======================== 显示训练配置 ========================
echo "========================================"
echo "训练配置信息："
echo "数据集: $dataset_name"
echo "配置文件: $config_name"
echo "使用GPU: $gpu"
echo "进程数: $np"
echo "CPU线程数: $omp"
echo "主端口: $master_port"
echo "========================================"

# 验证GPU是否可用
echo "验证GPU可用性..."
if ! nvidia-smi > /dev/null 2>&1; then
    echo "错误: nvidia-smi 不可用，请检查NVIDIA驱动"
    exit 1
fi

# 显示当前GPU状态
echo "当前GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | while IFS=, read -r idx name mem_used mem_total; do
    if [[ ",$gpu," == *",$idx,"* ]]; then
        echo "  GPU $idx: $name (使用中: ${mem_used}MB / ${mem_total}MB) ✓"
    fi
done

# 生成实验文件名
filename=$dataset_name"_$(date +%m%d_%H%M%S)"
echo "实验文件名: $filename"
echo "========================================"

# ======================== 启动训练 ========================
echo "启动多GPU训练..."
CUDA_VISIBLE_DEVICES=$gpu \
OMP_NUM_THREADS=$omp \
MASTER_PORT=$master_port \
torchrun --nproc_per_node=$np --master_port=$master_port \
train.py \
--config config/$config_name


# ======================== 单GPU训练配置（备用） ========================
# 如果需要单GPU训练，注释掉上面的多GPU部分，取消下面的注释：

# echo "启动单GPU训练..."
# gpu_single="4"  # 选择单个GPU
# CUDA_VISIBLE_DEVICES=$gpu_single \
# OMP_NUM_THREADS=$omp \
# torchrun --nproc_per_node=1 --master_port=$master_port \
# train.py \
# --config keshihua/config/$config_name



