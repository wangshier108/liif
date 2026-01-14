#!/bin/bash

INPUT_DIR=~/work/dataset/benchmark/B100/LR_x20
HR_DIR=~/work/dataset/benchmark/B100/HR
OUTPUT_DIR=~/work/b100/pred_x20_rdn_baseline
# OUTPUT_DIR=~/work/b100/pred_x20_rdn_ours
MODEL=~/work/rdn-liif.pth
# MODEL=~/final/liif/save/train_rdn_multi_samle_refinemlp21/epoch-last.pth

mkdir -p "$OUTPUT_DIR"

for input_file in "$INPUT_DIR"/*.png; do
    # 获取文件名（不带路径）
    filename=$(basename "$input_file")
    
    # 对应 HR 图像路径
    hr_file="$HR_DIR/$filename"

    # 使用 Python 获取 HR 图像分辨率
    size=$(python3 -c "
from PIL import Image
try:
    with Image.open('$hr_file') as img:
        print(f'{img.height},{img.width}')
except:
    exit(1)
")

    # 如果获取失败就跳过
    if [ $? -ne 0 ]; then
        echo "无法读取 HR 图像尺寸: $hr_file"
        continue
    fi

    # 输出路径
    output_file="$OUTPUT_DIR/$filename"
    echo "output: $output_file..."
    echo "Processing $filename with resolution $size ..."

    # 执行模型推理
    python demo.py \
        --input "$input_file" \
        --model "$MODEL" \
        --resolution "$size" \
        --output "$output_file"
done

echo "全部处理完成，结果保存在 $OUTPUT_DIR"
