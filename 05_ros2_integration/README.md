# ROS2视觉伺服系统

完整的ROS2视觉感知与伺服控制系统,适用于人形机器人抓取任务。

## 系统架构
```
[相机节点] → /camera/image_raw → [检测节点] → /detections → [视觉伺服]
                ↓                                    ↓
         [深度估计节点] ← /camera/depth      /cmd_vel → [机器人]
                ↓
         /object_pose_3d
```

## 功能特性

- ✅ 多种相机支持(USB/CSI/RTSP)
- ✅ YOLO实时物体检测
- ✅ 深度估计与3D定位
- ✅ PID视觉伺服控制
- ✅ 坐标系自动变换(tf2)
- ✅ 模块化设计,易于扩展

## 快速开始

### 1. 安装
```bash
# 克隆仓库
cd ~/ros2_ws/src
git clone <your-repo-url>

# 运行安装脚本
cd ~/ros2_ws
bash install.sh

# Source环境
source install/setup.bash
```

### 2. 测试相机
```bash
ros2 launch launch/test_camera.launch.py device_id:=0
```

### 3. 启动完整系统
```bash
ros2 launch launch/grasp_system.launch.py
```

### 4. 仅视觉部分(调试)
```bash
ros2 launch launch/vision_only.launch.py
```

## 包说明

### camera_publisher
相机发布节点,支持:
- USB/CSI/RTSP相机
- 畸变校正
- 图像压缩
- 相机参数发布

**话题:**
- `/camera/image_raw` - 原始图像
- `/camera/image_raw/compressed` - 压缩图像
- `/camera/camera_info` - 相机参数

### object_detector
YOLO物体检测节点

**话题:**
- 订阅: `/camera/image_raw`
- 发布: `/detections` - 检测结果
- 发布: `/detections/visualization` - 可视化图像

**参数:**
```yaml
model_path: 'yolov8n.pt'
confidence_threshold: 0.5
target_classes: ['apple', 'bottle']
```

### visual_servo
视觉伺服控制包,包含:
- PID控制器
- 深度估计
- 坐标变换

**话题:**
- 订阅: `/detections`, `/camera/depth`
- 发布: `/cmd_vel`, `/object_pose_3d`

## 参数配置

所有参数在`config/*.yaml`中配置,也可以通过Launch文件覆盖:
```python
Node(
    package='camera_publisher',
    executable='camera_node',
    parameters=[{
        'frame_rate': 60,
        'device_id': 1
    }]
)
```

## 常用命令
```bash
# 查看所有话题
ros2 topic list

# 查看图像
ros2 run rqt_image_view rqt_image_view

# 查看检测结果
ros2 topic echo /detections

# 查看tf树
ros2 run tf2_tools view_frames

# 录制数据
ros2 bag record -a
```

## 故障排查

### 相机无法打开
```bash
# 检查设备
ls -l /dev/video*
v4l2-ctl --list-devices

# 测试相机
ros2 launch launch/test_camera.launch.py
```

### 检测节点失败
```bash
# 检查模型路径
ls -l yolov8n.pt

# 测试YOLO
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### tf变换失败
```bash
# 查看tf树
ros2 run tf2_tools view_frames

# 检查变换
ros2 run tf2_ros tf2_echo base_link camera_optical_frame
```

## 开发指南

### 添加新节点

1. 创建新包:
```bash
cd src
ros2 pkg create my_node --build-type ament_python --dependencies rclpy
```

2. 编写节点代码

3. 更新`setup.py`

4. 构建:
```bash
colcon build --packages-select my_node
```

### 修改参数

编辑对应的`config/*.yaml`文件,然后重启节点。


## 贡献

欢迎提交Issue和PR!
