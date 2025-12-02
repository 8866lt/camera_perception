#!/bin/bash
# 批量转换YOLO模型到TensorRT引擎

set -e

# 配置
MODELS=("yolov8n" "yolov8s")
PRECISIONS=("fp32" "fp16")
IMG_SIZE=640
WORKSPACE=4

echo "========================================"
echo "批量转换YOLO模型到TensorRT"
echo "========================================"

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "处理模型: $MODEL"
    echo "----------------------------------------"
    
    # 检查PyTorch模型是否存在
    if [ ! -f "models/${MODEL}.pt" ]; then
        echo "下载预训练模型..."
        python -c "from ultralytics import YOLO; YOLO('${MODEL}.pt')"
        mv "${MODEL}.pt" models/
    fi
    
    # 导出ONNX
    if [ ! -f "models/${MODEL}.onnx" ]; then
        echo "导出ONNX..."
        python tensorrt_optimization/export_onnx.py \
            --weights models/${MODEL}.pt \
            --img-size $IMG_SIZE \
            --simplify \
            --output models/${MODEL}.onnx
    else
        echo "✓ ONNX已存在,跳过"
    fi
    
    # 构建不同精度的引擎
    for PRECISION in "${PRECISIONS[@]}"; do
        ENGINE="models/${MODEL}_${PRECISION}.engine"
        
        if [ -f "$ENGINE" ]; then
            echo "✓ ${PRECISION}引擎已存在,跳过"
            continue
        fi
        
        echo "构建${PRECISION}引擎..."
        
        if [ "$PRECISION" == "fp16" ]; then
            python tensorrt_optimization/build_engine.py \
                --onnx models/${MODEL}.onnx \
                --output $ENGINE \
                --fp16 \
                --workspace $WORKSPACE
        elif [ "$PRECISION" == "int8" ]; then
            # INT8需要校准数据
            if [ -d "calibration_images" ]; then
                python tensorrt_optimization/build_engine.py \
                    --onnx models/${MODEL}.onnx \
                    --output $ENGINE \
                    --int8 \
                    --calib-images calibration_images \
                    --workspace $WORKSPACE
            else
                echo "警告: 缺少校准图像,跳过INT8"
            fi
        else
            python tensorrt_optimization/build_engine.py \
                --onnx models/${MODEL}.onnx \
                --output $ENGINE \
                --workspace $WORKSPACE
        fi
    done
done

echo ""
echo "========================================"
echo "转换完成!"
echo "========================================"
echo ""
echo "生成的引擎文件:"
ls -lh models/*.engine

echo ""
echo "性能测试:"
echo "python tensorrt_optimization/benchmark.py --engines models/*.engine"
