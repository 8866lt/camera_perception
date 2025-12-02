# 基础检查
python calibration_checker.py --calib camera_calibration.yaml

# 完整检查(包括标定图像分析)
python calibration_checker.py \
    --calib camera_calibration.yaml \
    --images "calib_images/*.jpg" \
    --chessboard 9,6

# 测试畸变校正效果
python calibration_checker.py \
    --calib camera_calibration.yaml \
    --test-image test.jpg \
    --output-dir results/
```

---

## 更新GitHub仓库结构
```
02_camera_calibration/
├── README.md
├── generate_pattern.py          # 生成标定板
├── calibration_checker.py       # 标定质量检查
├── chessboard_calibration.py    # 之前的标定脚本
├── undistortion.py
├── examples/                     # 示例数据
│   ├── sample_calib.yaml        # 示例标定结果
│   └── sample_images/           # 示例标定图像
└── patterns/                     # ✅ 预生成的标定板PDF
    ├── chessboard_9x6_25mm.pdf  # ✅ 标准棋盘格
    ├── charuco_9x6_25mm.pdf     # ✅ ChArUco板
    └── circles_11x7_20mm.pdf    # ✅ 圆点阵
