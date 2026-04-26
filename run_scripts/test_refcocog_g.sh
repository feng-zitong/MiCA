#!/bin/bash

# =====================================================================
# MiCA REC (Referring Expression Comprehension) refcocog_g 测试脚本
# 测试 RefCOCOg Google 的 val
# =====================================================================

# 切换到项目根目录
cd "$(dirname "$0")/.."

# ======================== 测试配置 ========================
dataset_name="refcocog_g"
config_name="refcocog_g/MiCA_REC_base_g.yaml"
gpu="0"  # 使用的GPU

# 模型路径
model_path="/path/to/best_model.pth"  # set your checkpoint path

# ======================== 检查模型文件 ========================
if [ ! -f "$model_path" ]; then
    echo "❌ 错误: 模型文件不存在: $model_path"
    echo "请检查模型路径是否正确"
    exit 1
fi

echo "========================================"
echo "🎯 REC RefCOCOg Google 测试"
echo "========================================"
echo "模型路径: $model_path"
echo "配置文件: $config_name"
echo "使用GPU: $gpu"
echo "========================================"

# ======================== 按顺序测试 split ========================

# 测试 val
echo ""
echo "========== 测试 val =========="
CUDA_VISIBLE_DEVICES=$gpu \
python3 -u test.py \
--config config/$config_name \
--path $model_path \
--opts TEST.test_split val \
       TEST.test_lmdb datasets/lmdb/$dataset_name/val.lmdb \
       DATA.dataset $dataset_name \
       DATA.mask_root datasets/masks/$dataset_name
