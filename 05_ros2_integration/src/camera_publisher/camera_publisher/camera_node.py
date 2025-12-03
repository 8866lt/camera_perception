#!/usr/bin/env python3
"""
相机发布节点
支持USB/CSI/RTSP相机,发布原始图像和相机参数
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
from pathlib import Path

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        
        # 声明参数
        self.declare_parameter('device_id', 0)
        self.declare_parameter('frame_rate', 30)
        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 720)
        self.declare_parameter('camera_name', 'camera')
        self.declare_parameter('camera_info_path', '')
        self.declare_parameter('undistort', False)
        self.declare_parameter('publish_compressed', True)
        self.declare_parameter('jpeg_quality', 80)
        
        # 获取参数
        self.device_id = self.get_parameter('device_id').value
        self.frame_rate = self.get_parameter('frame_rate').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.camera_name = self.get_parameter('camera_name').value
        self.camera_info_path = self.get_parameter('camera_info_path').value
        self.undistort = self.get_parameter('undistort').value
        self.publish_compressed = self.get_parameter('publish_compressed').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value
        
        # 初始化
        self.bridge = CvBridge()
        self.frame_id = f'{self.camera_name}_optical_frame'
        
        # 打开相机
        self.cap = self._open_camera()
        
        # 加载标定参数
        self.camera_info = self._load_camera_info()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.new_camera_matrix = None
        self.roi = None
        
        if self.undistort and self.camera_info:
            self._setup_undistortion()
        
        # 创建发布者
        self.image_pub = self.create_publisher(
            Image,
            f'/{self.camera_name}/image_raw',
            10
        )
        
        self.camera_info_pub = self.create_publisher(
            CameraInfo,
            f'/{self.camera_name}/camera_info',
            10
        )
        
        if self.publish_compressed:
            self.compressed_pub = self.create_publisher(
                CompressedImage,
                f'/{self.camera_name}/image_raw/compressed',
                10
            )
        
        # 创建定时器
        timer_period = 1.0 / self.frame_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info(f'相机节点已启动: {self.camera_name}')
        self.get_logger().info(f'  分辨率: {self.width}x{self.height}')
        self.get_logger().info(f'  帧率: {self.frame_rate} Hz')
        self.get_logger().info(f'  畸变校正: {self.undistort}')
    
    def _open_camera(self):
        """打开相机"""
        # 检测是否为Jetson CSI相机
        if isinstance(self.device_id, str) and 'nvarguscamerasrc' in self.device_id:
            # CSI相机(GStreamer)
            gst_str = self.device_id
            cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        else:
            # USB相机
            cap = cv2.VideoCapture(self.device_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.frame_rate)
        
        if not cap.isOpened():
            self.get_logger().error(f'无法打开相机: {self.device_id}')
            raise RuntimeError('相机打开失败')
        
        # 验证分辨率
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_width != self.width or actual_height != self.height:
            self.get_logger().warn(
                f'请求分辨率 {self.width}x{self.height}, '
                f'实际分辨率 {actual_width}x{actual_height}'
            )
            self.width = actual_width
            self.height = actual_height
        
        return cap
    
    def _load_camera_info(self):
        """加载相机标定参数"""
        if not self.camera_info_path:
            self.get_logger().warn('未指定相机标定文件')
            return None
        
        calib_file = Path(self.camera_info_path)
        if not calib_file.exists():
            self.get_logger().warn(f'标定文件不存在: {self.camera_info_path}')
            return None
        
        try:
            with open(calib_file, 'r') as f:
                calib_data = yaml.safe_load(f)
            
            # 创建CameraInfo消息
            camera_info = CameraInfo()
            camera_info.header.frame_id = self.frame_id
            camera_info.width = calib_data['image_width']
            camera_info.height = calib_data['image_height']
            
            # 相机矩阵
            K = calib_data['camera_matrix']
            camera_info.k = [
                K[0], K[1], K[2],
                K[3], K[4], K[5],
                K[6], K[7], K[8]
            ]
            
            # 畸变系数
            D = calib_data['distortion_coefficients']
            camera_info.d = D
            camera_info.distortion_model = 'plumb_bob'
            
            # 投影矩阵
            camera_info.p = [
                K[0], 0.0,  K[2], 0.0,
                0.0,  K[4], K[5], 0.0,
                0.0,  0.0,  1.0,  0.0
            ]
            
            # 校正矩阵(单目为单位矩阵)
            camera_info.r = [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0
            ]
            
            self.get_logger().info(f'已加载标定参数: {self.camera_info_path}')
            return camera_info
        
        except Exception as e:
            self.get_logger().error(f'加载标定文件失败: {e}')
            return None
    
    def _setup_undistortion(self):
        """设置畸变校正"""
        if not self.camera_info:
            return
        
        # 提取参数
        K = np.array(self.camera_info.k).reshape(3, 3)
        D = np.array(self.camera_info.d)
        
        self.camera_matrix = K
        self.dist_coeffs = D
        
        # 计算最优新相机矩阵
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeffs,
            (self.width, self.height),
            alpha=0,  # 裁剪黑边
            newImgSize=(self.width, self.height)
        )
        
        self.get_logger().info('畸变校正已启用')
    
    def timer_callback(self):
        """定时器回调"""
        ret, frame = self.cap.read()
        
        if not ret:
            self.get_logger().warn('读取帧失败')
            return
        
        # 畸变校正
        if self.undistort and self.camera_matrix is not None:
            frame = cv2.undistort(
                frame,
                self.camera_matrix,
                self.dist_coeffs,
                None,
                self.new_camera_matrix
            )
            
            # 裁剪ROI
            x, y, w, h = self.roi
            if w > 0 and h > 0:
                frame = frame[y:y+h, x:x+w]
        
        # 创建Header
        timestamp = self.get_clock().now().to_msg()
        
        # 发布原始图像
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        img_msg.header.stamp = timestamp
        img_msg.header.frame_id = self.frame_id
        self.image_pub.publish(img_msg)
        
        # 发布压缩图像
        if self.publish_compressed:
            _, buffer = cv2.imencode(
                '.jpg',
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )
            
            compressed_msg = CompressedImage()
            compressed_msg.header.stamp = timestamp
            compressed_msg.header.frame_id = self.frame_id
            compressed_msg.format = 'jpeg'
            compressed_msg.data = buffer.tobytes()
            self.compressed_pub.publish(compressed_msg)
        
        # 发布相机参数
        if self.camera_info:
            self.camera_info.header.stamp = timestamp
            self.camera_info_pub.publish(self.camera_info)
    
    def destroy_node(self):
        """清理资源"""
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = CameraPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'错误: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
