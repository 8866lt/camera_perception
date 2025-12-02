# RGB相机工程实践完整指南

<p align="center">
  <img src="docs/images/camera_pipeline.png" alt="相机处理流水线" width="800"/>
</p>

## 📋 目录

- [项目简介](#项目简介)
- [硬件要求](#硬件要求)
- [环境配置](#环境配置)
  - [方式一: Docker部署(推荐)](#方式一-docker部署推荐)
  - [方式二: 本地环境配置](#方式二-本地环境配置)
- [快速开始](#快速开始)
- [模块详解](#模块详解)
- [性能优化](#性能优化)
- [常见问题](#常见问题)
- [参考资料](#参考资料)

---

## 项目简介

本项目提供人形机器人RGB相机感知的完整工程实践,涵盖从相机标定到深度学习推理的全流程。

**主要功能:**
- ✅ 多种相机支持(USB/CSI/RTSP)
- ✅ 相机标定与畸变校正
- ✅ YOLO物体检测 + TensorRT加速
- ✅ 人体姿态估计(MediaPipe)
- ✅ ROS2无缝集成
- ✅ 视觉伺服应用示例

**适用平台:**
- x86_64 Linux (Ubuntu 20.04/22.04)
- NVIDIA Jetson (Nano/NX/AGX Orin)
- Raspberry Pi 4 (部分功能)

---

## 硬件要求

### 最低配置

| 组件 | 要求 |
|-----|------|
| CPU | Intel i5 或同等性能 |
| 内存 | 8GB RAM |
| GPU | NVIDIA GPU (可选,用于加速) |
| 相机 | USB 2.0相机 或 CSI相机 |
| 存储 | 20GB 可用空间 |

### 推荐配置(Jetson平台)

| 型号 | 性能 | 适用场景 |
|-----|------|---------|
| Jetson Nano | 入门级 | 学习、原型验证 |
| Jetson Xavier NX | 中端 | 实时检测、SLAM |
| Jetson AGX Orin | 高端 | 多相机、高帧率 |

### 支持的相机

**USB相机:**
- 罗技 C920/C930e
- 任何标准UVC相机

**工业相机:**
- Basler ace系列
- FLIR Blackfly

**深度相机:**
- Intel RealSense D435i
- Azure Kinect

**Jetson CSI相机:**
- Raspberry Pi Camera Module V2
- IMX219/IMX477传感器

---

## 环境配置

### 方式一: Docker部署(推荐)

Docker方式提供开箱即用的环境,避免依赖冲突。

#### 1. 安装Docker

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# Jetson平台(使用NVIDIA官方脚本)
# 参考: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

#### 2. 构建Docker镜像

**x86平台:**

```bash
cd docker
docker build -f Dockerfile.x86 -t camera-perception:x86 .
```

**Jetson平台:**

```bash
cd docker
docker build -f Dockerfile.jetson -t camera-perception:jetson .
```

#### 3. 运行容器

**x86平台(带GPU):**

```bash
docker run --gpus all \
  --rm -it \
  --privileged \
  -v /dev:/dev \
  -v $PWD:/workspace \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --net=host \
  camera-perception:x86 \
  bash
```

**Jetson平台:**

```bash
docker run --runtime nvidia \
  --rm -it \
  --privileged \
  -v /dev:/dev \
  -v $PWD:/workspace \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --net=host \
  camera-perception:jetson \
  bash
```

**Mac/Windows用户:**

由于相机访问限制,建议使用虚拟机或双系统。

---

### 方式二: 本地环境配置

#### 1. 系统要求

```bash
# 检查系统版本
lsb_release -a

# 支持: Ubuntu 20.04/22.04
```

#### 2. 安装Python依赖

**创建虚拟环境:**

```bash
# 安装venv
sudo apt-get update
sudo apt-get install python3-venv python3-dev

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 升级pip
pip install --upgrade pip
```

**安装基础依赖:**

```bash
pip install -r requirements.txt
```

**requirements.txt 内容:**

```txt
# 基础库
numpy>=1.21.0
opencv-python>=4.5.0
opencv-contrib-python>=4.5.0
PyYAML>=5.4
Pillow>=8.0

# 深度学习
torch>=1.10.0
torchvision>=0.11.0
onnx>=1.10.0
onnxruntime>=1.10.0  # CPU版本
# onnxruntime-gpu>=1.10.0  # GPU版本,与上面二选一

# 姿态估计
mediapipe>=0.8.9

# ROS2 (如果使用ROS2)
# 不通过pip安装,使用apt安装ROS2后,source setup.bash

# 工具
tqdm>=4.62.0
matplotlib>=3.4.0
```

**Jetson平台特殊依赖:**

```bash
# TensorRT (Jetson已预装,只需Python绑定)
pip install pycuda

# Jetson特定优化
pip install jetson-stats
```

#### 3. 安装OpenCV (完整版)

系统自带的OpenCV可能缺少某些模块,推荐从源码编译:

```bash
cd scripts
bash setup_opencv.sh
```

**setup_opencv.sh 内容:**

```bash
#!/bin/bash
# OpenCV编译脚本 (支持CUDA、GStreamer)

set -e

OPENCV_VERSION=4.8.0
BUILD_DIR=/tmp/opencv_build

echo "开始编译 OpenCV ${OPENCV_VERSION}..."

# 安装依赖
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git pkg-config \
    libjpeg-dev libtiff-dev libpng-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev gfortran \
    python3-dev python3-numpy

# 下载源码
mkdir -p $BUILD_DIR && cd $BUILD_DIR
git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git
git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git

# 创建编译目录
cd opencv && mkdir build && cd build

# 检测CUDA
if command -v nvcc &> /dev/null; then
    CUDA_ARCH_BIN=""
    
    # 检测GPU架构
    if lspci | grep -i nvidia | grep -qi "jetson"; then
        # Jetson设备
        if nvidia-smi | grep -qi "Orin"; then
            CUDA_ARCH_BIN="8.7"
        elif nvidia-smi | grep -qi "Xavier"; then
            CUDA_ARCH_BIN="7.2"
        else
            CUDA_ARCH_BIN="5.3"  # Nano
        fi
    else
        # 桌面GPU,自动检测
        CUDA_ARCH_BIN="6.0,6.1,7.0,7.5,8.0,8.6"
    fi
    
    WITH_CUDA=ON
    echo "检测到CUDA,架构: ${CUDA_ARCH_BIN}"
else
    WITH_CUDA=OFF
    echo "未检测到CUDA,将编译CPU版本"
fi

# CMake配置
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D WITH_CUDA=${WITH_CUDA} \
    -D CUDA_ARCH_BIN="${CUDA_ARCH_BIN}" \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D WITH_GSTREAMER=ON \
    -D WITH_V4L=ON \
    -D WITH_QT=OFF \
    -D WITH_OPENGL=ON \
    -D BUILD_opencv_python3=ON \
    -D PYTHON3_EXECUTABLE=$(which python3) \
    -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    ..

# 编译 (使用所有CPU核心)
NPROC=$(nproc)
echo "使用 ${NPROC} 个核心编译..."
make -j${NPROC}

# 安装
sudo make install
sudo ldconfig

# 验证
python3 -c "import cv2; print(f'OpenCV {cv2.__version__} 安装成功')"
python3 -c "import cv2; print(f'CUDA: {cv2.cuda.getCudaEnabledDeviceCount() > 0}')"

echo "OpenCV编译完成!"
```

**使用方法:**

```bash
chmod +x scripts/setup_opencv.sh
./scripts/setup_opencv.sh

# 编译时间: 
# - Jetson Nano: 2-3小时
# - Jetson Xavier NX: 30-60分钟  
# - x86 (8核): 15-30分钟
```

#### 4. 安装TensorRT (Jetson)

Jetson设备已预装TensorRT,只需安装Python绑定:

```bash
# 查看TensorRT版本
dpkg -l | grep TensorRT

# 安装Python绑定
pip install pycuda
```

**x86平台安装TensorRT:**

```bash
cd scripts
bash setup_tensorrt.sh
```

**setup_tensorrt.sh 内容:**

```bash
#!/bin/bash
# TensorRT安装脚本 (x86平台)

set -e

TRT_VERSION=8.5.3.1
CUDA_VERSION=11.8

echo "安装 TensorRT ${TRT_VERSION}..."

# 下载TensorRT (需要NVIDIA账号)
echo "请从以下链接下载TensorRT:"
echo "https://developer.nvidia.com/nvidia-tensorrt-8x-download"
echo ""
echo "选择: TensorRT ${TRT_VERSION} for Linux x86_64 and CUDA ${CUDA_VERSION}"
echo ""
read -p "下载完成后,输入tar.gz文件路径: " TRT_TAR

# 解压
TRT_DIR=$(basename ${TRT_TAR} .tar.gz)
tar -xzf ${TRT_TAR}

# 安装
cd ${TRT_DIR}/python
pip install tensorrt-*-cp3*-none-linux_x86_64.whl

cd ../onnx_graphsurgeon
pip install onnx_graphsurgeon-*-py2.py3-none-any.whl

# 设置环境变量
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/../lib" >> ~/.bashrc
source ~/.bashrc

# 验证
python3 -c "import tensorrt; print(f'TensorRT {tensorrt.__version__} 安装成功')"

echo "TensorRT安装完成!"
```

#### 5. 安装ROS2 (可选)

如果需要ROS2集成功能:

```bash
# Ubuntu 22.04 - ROS2 Humble
sudo apt update && sudo apt install -y curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo apt-key add -
sudo sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'
sudo apt update
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions

# 设置环境
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# 安装vision相关包
sudo apt install -y \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-compressed-image-transport \
    ros-humble-vision-msgs
```

**Ubuntu 20.04用户:**

```bash
# 使用ROS2 Foxy
sudo apt install -y ros-foxy-desktop
```

#### 6. 相机权限配置

**USB相机权限:**

```bash
# 添加用户到video组
sudo usermod -aG video $USER

# 创建udev规则
sudo tee /etc/udev/rules.d/99-camera.rules > /dev/null <<EOF
SUBSYSTEM=="video4linux", GROUP="video", MODE="0660"
EOF

# 重新加载规则
sudo udevadm control --reload-rules
sudo udevadm trigger

# 重新登录生效
```

**CSI相机(Jetson):**

```bash
# 检查CSI相机是否被识别
ls -l /dev/video*

# 测试CSI相机
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1920,height=1080' ! nvoverlaysink
```

---

## 快速开始

### 1. 测试相机连接

```bash
# 激活虚拟环境(如果使用)
source venv/bin/activate

# USB相机测试
python 01_camera_basics/camera_capture.py --source 0

# 预期输出:
# 相机已打开:
#   分辨率: 640x480
#   帧率: 30.0
# 
# 操作说明:
#   q - 退出
#   s - 保存当前帧
#   f - 显示/隐藏帧率
```

**常见问题排查:**

```bash
# 1. 列出所有视频设备
v4l2-ctl --list-devices

# 2. 查看相机支持的格式
v4l2-ctl -d /dev/video0 --list-formats-ext

# 3. 测试相机(使用ffplay)
ffplay /dev/video0
```

### 2. 相机标定(5分钟快速标定)

**准备工作:**

1. 打印标定板:使用`02_camera_calibration/generate_pattern.py`生成

```bash
python 02_camera_calibration/generate_pattern.py \
    --type chessboard \
    --cols 9 \
    --rows 6 \
    --size 25 \
    --output calibration_board.pdf

# 打印到A4纸,测量实际格子尺寸(应该是25mm)
```

2. 固定标定板:贴在硬纸板或亚克力板上,保持平整

**标定流程:**

```bash
# 步骤1: 采集图像(20-30张)
python 02_camera_calibration/chessboard_calibration.py \
    --mode capture \
    --camera 0

# 操作:
# - 移动标定板到不同位置(左/右/上/下/中心)
# - 旋转标定板到不同角度(0°/30°/45°/60°)
# - 改变标定板距离(近/中/远)
# - 看到"Detected! Press 's' to save"时按's'保存
# - 采集20-30张后按'q'退出

# 步骤2: 执行标定
python 02_camera_calibration/chessboard_calibration.py \
    --mode calibrate \
    --images "calib_images/*.jpg" \
    --output camera_calib.yaml

# 预期输出:
# 标定成功!
# 重投影误差(RMS): 0.3521 像素  # <0.5像素为优秀
# 
# 相机内参矩阵:
# [[635.2  0.0  318.4]
#  [  0.0 636.1 241.2]
#  [  0.0   0.0   1.0]]
# 
# 视场角(FOV):
#   水平: 62.3°
#   垂直: 48.7°

# 步骤3: 测试畸变校正
python 02_camera_calibration/chessboard_calibration.py \
    --mode test \
    --camera 0 \
    --output camera_calib.yaml

# 观察左右对比图,直线应该变直
```

**标定质量评估:**

| 重投影误差 | 质量 | 说明 |
|----------|-----|------|
| < 0.3像素 | 优秀 | 可用于精密测量 |
| 0.3-0.5像素 | 良好 | 适合大多数应用 |
| 0.5-1.0像素 | 可接受 | 一般应用足够 |
| > 1.0像素 | 较差 | 需重新标定 |

**提高标定质量的技巧:**

- ✅ 采集更多图像(30张以上)
- ✅ 覆盖整个视野范围
- ✅ 包含各种角度和距离
- ✅ 保证标定板平整
- ✅ 光照均匀,避免过曝和阴影
- ❌ 避免模糊图像
- ❌ 避免标定板填满整个画面

### 3. 物体检测 + TensorRT加速

**准备YOLO模型:**

```bash
cd 04_object_detection/tensorrt_optimization

# 步骤1: 下载预训练模型
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt

# 步骤2: 导出ONNX
python onnx_export.py \
    --weights yolov5s.pt \
    --img-size 640 \
    --batch-size 1 \
    --output yolov5s.onnx

# 步骤3: 构建TensorRT引擎
python tensorrt_build.py \
    --onnx yolov5s.onnx \
    --engine yolov5s.engine \
    --precision fp16  # Jetson使用fp16, 桌面GPU可用fp32

# 构建时间:
# - Jetson Nano: 5-10分钟
# - Jetson Xavier NX: 2-3分钟
# - RTX 3060: 30-60秒
```

**运行检测:**

```bash
cd 04_object_detection

# 实时检测
python yolo_tensorrt.py \
    --engine tensorrt_optimization/yolov5s.engine \
    --camera 0 \
    --conf 0.5

# 预期性能:
# - Jetson Nano: 10-15 FPS
# - Jetson Xavier NX: 30-40 FPS
# - Jetson AGX Orin: 60+ FPS
# - RTX 3060: 100+ FPS
```

**性能对比:**

| 平台 | 纯PyTorch | TensorRT | 加速比 |
|-----|----------|----------|--------|
| Jetson Nano | 3 FPS | 12 FPS | 4x |
| Jetson Xavier NX | 8 FPS | 35 FPS | 4.4x |
| RTX 3060 | 45 FPS | 120 FPS | 2.7x |

### 4. 人体姿态估计

```bash
cd 05_pose_estimation

# MediaPipe姿态估计
python mediapipe_pose.py --camera 0

# 手部追踪
python hand_tracking.py --camera 0

# 人脸关键点
python face_landmarks.py --camera 0
```

**MediaPipe性能:**

- CPU模式: 30-60 FPS (取决于CPU性能)
- GPU加速: 60+ FPS (需要GPU)

### 5. ROS2集成

```bash
# 确保已source ROS2环境
source /opt/ros/humble/setup.bash

cd 06_ros2_integration

# 启动相机发布节点
python camera_publisher.py \
    --camera 0 \
    --topic /camera/image_raw \
    --camera-info camera_calib.yaml

# 在另一个终端查看话题
ros2 topic list
ros2 topic hz /camera/image_raw
ros2 run rqt_image_view rqt_image_view
```

---

## 模块详解

### 模块1: 相机基础操作

**文件:** `01_camera_basics/camera_capture.py`

**功能:**
- 支持USB/CSI/RTSP相机
- 分辨率和帧率设置
- 实时FPS显示
- 图像保存

**核心代码解析:**

```python
# USB相机
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Jetson CSI相机 (使用GStreamer)
gst_str = (
    f"nvarguscamerasrc ! "
    f"video/x-raw(memory:NVMM), width=1920, height=1080 ! "
    f"nvvidconv ! video/x-raw, format=BGRx ! "
    f"videoconvert ! video/x-raw, format=BGR ! appsink"
)
cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
```

**性能优化技巧:**

```python
# 1. 使用硬件加速 (Jetson)
# 在GStreamer管道中使用nvvidconv

# 2. 减少不必要的拷贝
ret, frame = cap.read()  # 直接使用,不要拷贝

# 3. 降低分辨率
# 640x480足够大多数应用,比1080p快4倍
```

---

### 模块2: 相机标定

**文件:** `02_camera_calibration/chessboard_calibration.py`

**原理:**

相机成像模型:
```
世界坐标系 → 相机坐标系 → 图像坐标系

[u]   [fx  0  cx]   [X]
[v] = [ 0 fy cy] * [Y]
[1]   [ 0  0  1]   [Z]

其中:
- (fx, fy): 焦距(像素单位)
- (cx, cy): 主点(光轴与图像平面交点)
- (X, Y, Z): 相机坐标系中的3D点
- (u, v): 图像像素坐标
```

**畸变模型:**

```
径向畸变: 
  x' = x(1 + k1*r² + k2*r⁴ + k3*r⁶)
  y' = y(1 + k1*r² + k2*r⁴ + k3*r⁶)

切向畸变:
  x' = x + 2p1*xy + p2(r² + 2x²)
  y' = y + p1(r² + 2y²) + 2p2*xy

其中 r² = x² + y²
```

**标定板选择:**

| 类型 | 优势 | 劣势 | 适用场景 |
|-----|------|------|---------|
| 棋盘格 | 精度高,易打印 | 对称性易混淆 | 通用标定 |
| ChArUco | 抗遮挡,无歧义 | 需要打印精度高 | 推荐使用 |
| 圆点阵 | 亚像素精度最高 | 打印要求高 | 高精度测量 |

---

### 模块3: 图像处理

**文件:** `03_image_processing/undistortion.py`

**畸变校正原理:**

```python
# 方法1: 直接校正(慢)
dst = cv2.undistort(src, camera_matrix, dist_coeffs)

# 方法2: 查找表法(快,推荐实时应用)
map1, map2 = cv2.initUndistortRectifyMap(
    camera_matrix, dist_coeffs, None, 
    new_camera_matrix, img_size, cv2.CV_16SC2
)
dst = cv2.remap(src, map1, map2, cv2.INTER_LINEAR)

# 性能对比:
# - undistort: 每帧重新计算,约8ms (640x480)
# - remap: 使用预计算映射表,约2ms (640x480)
```

**自动曝光优化:**

```python
# 文件: 03_image_processing/auto_exposure.py

# 直方图均衡化
def auto_exposure_histogram(image):
    # 转换到YUV空间
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # 对Y通道(亮度)进行直方图均衡化
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    # 转回BGR
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

# CLAHE (对比度受限的自适应直方图均衡化)
def auto_exposure_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
```

---

### 模块4: YOLO检测 + TensorRT

**文件:** `04_object_detection/yolo_tensorrt.py`

**TensorRT优化原理:**

1. **算子融合**: 合并多个层减少内存访问
2. **精度校准**: FP16/INT8量化
3. **内核调优**: 为特定硬件选择最优kernel

**模型转换流程:**

```
PyTorch (.pt) 
    ↓ [export]
ONNX (.onnx)
    ↓ [build engine]
TensorRT (.engine)
    ↓ [inference]
结果
```

**精度选择:**

| 精度 | 速度 | 精度损失 | 适用平台 |
|-----|------|---------|---------|
| FP32 | 基准 | 0% | 桌面GPU |
| FP16 | 2x | <1% | Jetson,现代GPU |
| INT8 | 4x | 1-3% | 需要校准数据集 |

**INT8量化示例:**

```python
# 文件: 04_object_detection/tensorrt_optimization/tensorrt_build.py

def build_engine_int8(onnx_path, calib_dataset):
    # 创建校准器
    calibrator = Int8Calibrator(calib_dataset)
    
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator
    
    # 构建引擎
    engine = builder.build_engine(network, config)
    return engine
```

---

### 模块5: 姿态估计

**文件:** `05_pose_estimation/mediapipe_pose.py`

**MediaPipe Pose关键点:**

```
0: nose (鼻子)
1-2: left/right eye (左右眼)
3-4: left/right ear (左右耳)
5-6: left/right shoulder (左右肩)
7-8: left/right elbow (左右肘)
9-10: left/right wrist (左右腕)
11-12: left/right hip (左右髋)
13-14: left/right knee (左右膝)
15-16: left/right ankle (左右踝)
```

**应用示例: 跌倒检测**

```python
def detect_fall(landmarks):
    """
    简单的跌倒检测算法
    判断依据: 躯干角度
    """
    # 获取关键点
    shoulder = landmarks[5]  # 左肩
    hip = landmarks[11]      # 左髋
    
    # 计算躯干角度
    angle = np.arctan2(hip.y - shoulder.y, 
                      hip.x - shoulder.x)
    angle_deg = np.degrees(angle)
    
    # 躯干接近水平 → 可能跌倒
    if abs(angle_deg) < 30:  # 躯干与水平夹角<30°
        return True
    return False
```

---

### 模块6: ROS2集成

**文件:** `06_ros2_integration/camera_publisher.py`

**ROS2相机发布节点:**

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        
        # 创建发布者
        self.image_pub = self.create_publisher(
            Image, '/camera/image_raw', 10)
        self.info_pub = self.create_publisher(
            CameraInfo, '/camera/camera_info', 10)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 定时器(30Hz)
        self.timer = self.create_timer(1.0/30.0, self.timer_callback)
        
        # 打开相机
        self.cap = cv2.VideoCapture(0)
    
    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # 转换为ROS Image消息
            msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera'
            
            # 发布
            self.image_pub.publish(msg)
```

**Launch文件示例:**

```python
# 06_ros2_integration/launch/camera.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='camera_perception',
            executable='camera_publisher',
            name='camera',
            parameters=[
                {'camera_id': 0},
                {'frame_rate': 30},
                {'image_width': 640},
                {'image_height': 480}
            ]
        ),
        Node(
            package='image_proc',
            executable='image_proc',
            name='image_proc',
            remappings=[
                ('image', '/camera/image_raw')
            ]
        )
    ])
```

---

## 性能优化

### 1. 减少延迟

**相机配置:**

```python
# 减少缓冲区(降低延迟但可能丢帧)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 禁用自动对焦(避免对焦延迟)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 30)  # 手动对焦值

# 禁用自动曝光
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 手动模式
cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # 曝光值
```

**多线程捕获:**

```python
import threading
from queue import Queue

class CameraThread(threading.Thread):
    def __init__(self, camera_id):
        super().__init__()
        self.cap = cv2.VideoCapture(camera_id)
        self.queue = Queue(maxsize=1)
        self.running = True
    
    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not self.queue.full():
                self.queue.put(frame)
    
    def read(self):
        return self.queue.get()
```

### 2. 提高帧率

**降低分辨率:**

```python
# 640x480 vs 1920x1080
# - 像素数少9倍
# - 处理速度提升约9倍
# - 传输带宽少9倍
```

**硬件加速(Jetson):**

```python
# 使用NVMM(零拷贝)
gst_str = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! appsink"
)
```

**使用GPU加速OpenCV:**

```python
# 上传到GPU
gpu_frame = cv2.cuda_GpuMat()
gpu_frame.upload(frame)

# GPU上处理
gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
gpu_blur = cv2.cuda.GaussianBlur(gpu_gray, (5,5), 0)

# 下载回CPU
result = gpu_blur.download()
```

### 3. 降低功耗(Jetson)

```bash
# 查看当前模式
sudo nvpmodel -q

# 设置为省电模式
sudo nvpmodel -m 1

# 限制最大功率(例如10W)
sudo nvpmodel -m 0
sudo jetson_clocks --show
```

**代码层面优化:**

```python
# 动态调整处理频率
def adaptive_processing(frame, fps_target=30):
    # 只在帧率低于目标时跳过处理
    if current_fps < fps_target:
        return None  # 跳过这一帧
    else:
        return process_frame(frame)
```

---

## 常见问题

### Q1: 相机无法打开

**症状:**
```
cv2.error: (-215) !_src.empty() in function 'cvtColor'
```

**排查步骤:**

```bash
# 1. 检查设备是否存在
ls -l /dev/video*

# 2. 检查权限
sudo chmod 666 /dev/video0

# 3. 检查是否被占用
lsof /dev/video0

# 4. 测试基础捕获
ffmpeg -f v4l2 -i /dev/video0 -frames 1 test.jpg
```

**解决方案:**

```python
# 尝试不同的backend
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Linux
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # macOS
```

---

### Q2: 标定重投影误差过大

**原因:**
- 标定板不平整
- 采集图像模糊
- 标定板角度范围不够
- 图像数量太少

**解决:**

```bash
# 1. 检查标定图像质量
python scripts/check_calibration_quality.py --images "calib_images/*.jpg"

# 2. 重新采集
# - 至少20张图像
# - 覆盖画面的中心、四角、边缘
# - 包含近距离(30cm)和远距离(2m)
# - 旋转角度: 0°, 30°, 45°, 60°

# 3. 使用ChArUco板(更鲁棒)
python 02_camera_calibration/charuco_calibration.py
```

---

### Q3: TensorRT引擎构建失败

**症状:**
```
[TensorRT] ERROR: engine.cpp (1047) - Serialization Error in validate: 0
```

**原因:**
- ONNX模型不兼容
- TensorRT版本不匹配
- 算子不支持

**解决:**

```bash
# 1. 检查ONNX模型
python -c "import onnx; model = onnx.load('model.onnx'); onnx.checker.check_model(model)"

# 2. 简化模型
python -m onnxsim model.onnx model_sim.onnx

# 3. 使用兼容的版本
# PyTorch 1.12 + TensorRT 8.5 (Jetson)
# PyTorch 2.0 + TensorRT 8.6 (Desktop)

# 4. 逐层检查
trtexec --onnx=model.onnx --verbose
```

---

### Q4: ROS2图像传输延迟高

**原因:**
- 使用未压缩图像传输
- 网络带宽不足
- QoS设置不当

**解决:**

```python
# 1. 使用压缩图像传输
from sensor_msgs.msg import CompressedImage

# 发布压缩图像
compressed_msg = self.bridge.cv2_to_compressed_imgmsg(frame, dst_format='jpg')
self.pub.publish(compressed_msg)

# 2. 调整QoS
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,  # 允许丢包
    history=HistoryPolicy.KEEP_LAST,
    depth=1  # 只保留最新一帧
)

self.pub = self.create_publisher(Image, '/camera/image', qos)

# 3. 降低分辨率或帧率
```

---

### Q5: Jetson运行缓慢

**排查:**

```bash
# 1. 检查当前状态
jtop  # 查看CPU/GPU/内存使用率

# 2. 检查温度
cat /sys/devices/virtual/thermal/thermal_zone*/temp

# 3. 检查电源模式
sudo nvpmodel -q

# 4. 检查是否降频
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
```

**优化:**

```bash
# 1. 最大性能模式
sudo nvpmodel -m 0
sudo jetson_clocks

# 2. 关闭图形界面(命令行模式)
sudo systemctl set-default multi-user.target
sudo reboot

# 3. 增加swap(如果内存不足)
sudo fallocate -l 4G /var/swapfile
sudo chmod 600 /var/swapfile
sudo mkswap /var/swapfile
sudo swapon /var/swapfile
```

---

### Q6: MediaPipe在Jetson上无法安装

**问题:**
```
ERROR: Could not find a version that satisfies the requirement mediapipe
```

**解决:**

```bash
# 方法1: 使用预编译wheel (推荐)
wget https://github.com/PINTO0309/mediapipe-bin/releases/download/v0.8.11/mediapipe-0.8.11_cuda11.4-cp38-cp38-linux_aarch64.whl
pip install mediapipe-0.8.11_cuda11.4-cp38-cp38-linux_aarch64.whl

# 方法2: 从源码编译 (耗时2-3小时)
git clone https://github.com/google/mediapipe.git
cd mediapipe
# 参考官方文档编译
```

---

## Docker镜像配置

**Dockerfile.x86:**

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 安装基础依赖
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libopencv-dev \
    python3-opencv \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    v4l-utils \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# 设置工作目录
WORKDIR /workspace

# 设置环境变量
ENV PYTHONPATH=/workspace
ENV DISPLAY=:0

CMD ["/bin/bash"]
```

**Dockerfile.jetson:**

```dockerfile
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# 安装额外依赖
RUN apt-get update && apt-get install -y \
    python3-opencv \
    libopencv-dev \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# 安装Python包
RUN pip3 install --no-cache-dir \
    pycuda \
    mediapipe \
    PyYAML \
    tqdm

WORKDIR /workspace

CMD ["/bin/bash"]
```

**构建和使用:**

```bash
# 构建
docker build -f docker/Dockerfile.x86 -t camera-perception:x86 .

# 运行(启用相机和显示)
docker run --rm -it \
    --gpus all \
    --privileged \
    -v /dev:/dev \
    -v $PWD:/workspace \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --net=host \
    camera-perception:x86
```

---

## 性能基准测试

使用提供的基准测试脚本:

```bash
python scripts/benchmark.py --all

# 输出示例:
# ========== 性能基准测试 ==========
# 平台: Jetson Xavier NX
# 
# 1. 相机捕获 (640x480@30fps)
#    - 平均延迟: 33.2ms
#    - 稳定性: 99.2%
# 
# 2. 畸变校正
#    - 平均耗时: 2.1ms
#    - 吞吐量: 476 fps
# 
# 3. YOLO检测 (TensorRT FP16)
#    - 平均耗时: 28.5ms
#    - 吞吐量: 35 fps
# 
# 4. 姿态估计 (MediaPipe)
#    - 平均耗时: 16.3ms
#    - 吞吐量: 61 fps
```

---

## 参考资料

**官方文档:**
- [OpenCV Documentation](https://docs.opencv.org/)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/)
- [MediaPipe](https://google.github.io/mediapipe/)
- [ROS2 Documentation](https://docs.ros.org/en/humble/)

**推荐阅读:**
- Zhang, Z. "A flexible new technique for camera calibration." (2000)
- Redmon, J. "YOLOv3: An Incremental Improvement." (2018)

**视频教程:**
- [相机标定原理](https://www.youtube.com/watch?v=...)
- [TensorRT优化实战](https://www.youtube.com/watch?v=...)

**社区资源:**
- [OpenCV论坛](https://forum.opencv.org/)
- [NVIDIA开发者论坛](https://forums.developer.nvidia.com/)
- [ROS Answers](https://answers.ros.org/)

---

## 贡献指南

欢迎提交Issue和Pull Request!

**开发环境设置:**

```bash
# Fork并克隆仓库
git clone https://github.com/8866lt/camera-perception.git
cd camera-perception

# 创建分支
git checkout -b feature/your-feature

# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
pytest tests/

# 提交代码
git commit -am "Add your feature"
git push origin feature/your-feature
```

---

## 联系方式

- GitHub: [(https://github.com/8866lt)]
- 知乎: [https://www.zhihu.com/people/su-xin-ran-64-35)]
- Email: [hehuaizhou@foxmail.com]

---

**最后更新:** 2025年12月

**版本:** v1.0.0


