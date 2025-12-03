#!/usr/bin/env python3
"""
物体检测节点
使用YOLO进行实时物体检测
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

class ObjectDetectorNode(Node):
    def __init__(self):
        super().__init__('object_detector')
        
        # 声明参数
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('target_classes', [])
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('publish_visualization', True)
        
        # 获取参数
        self.model_path = self.get_parameter('model_path').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.target_classes = self.get_parameter('target_classes').value
        self.device = self.get_parameter('device').value
        self.publish_viz = self.get_parameter('publish_visualization').value
        
        # 初始化
        self.bridge = CvBridge()
        
        if not ULTRALYTICS_AVAILABLE:
            self.get_logger().error('ultralytics未安装! pip install ultralytics')
            raise ImportError('ultralytics not available')
        
        # 加载模型
        self.get_logger().info(f'加载模型: {self.model_path}')
        self.model = YOLO(self.model_path)
        
        # 预热
        self.get_logger().info('预热模型...')
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model(dummy, verbose=False)
        self.get_logger().info('模型预热完成')
        
        # 创建订阅者
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # 创建发布者
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/detections',
            10
        )
        
        if self.publish_viz:
            self.viz_pub = self.create_publisher(
                Image,
                '/detections/visualization',
                10
            )
        
        # 统计
        self.frame_count = 0
        self.detection_count = 0
        
        self.get_logger().info('检测节点已启动')
        self.get_logger().info(f'  模型: {self.model_path}')
        self.get_logger().info(f'  置信度阈值: {self.confidence_threshold}')
        self.get_logger().info(f'  设备: {self.device}')
    
    def image_callback(self, msg):
        """图像回调"""
        self.frame_count += 1
        
        # 转为OpenCV格式
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {e}')
            return
        
        # YOLO检测
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )[0]
        
        # 解析结果
        detections = Detection2DArray()
        detections.header = msg.header
        
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                # 提取信息
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                
                # 过滤目标类别
                if self.target_classes and cls_name not in self.target_classes:
                    continue
                
                # 创建Detection2D消息
                detection = Detection2D()
                
                # 边界框中心和尺寸
                x1, y1, x2, y2 = xyxy
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                detection.bbox.center.position.x = float(center_x)
                detection.bbox.center.position.y = float(center_y)
                detection.bbox.size_x = float(width)
                detection.bbox.size_y = float(height)
                
                # 类别和置信度
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = cls_name
                hypothesis.hypothesis.score = conf
                detection.results.append(hypothesis)
                
                detections.detections.append(detection)
                self.detection_count += 1
        
        # 发布检测结果
        self.detection_pub.publish(detections)
        
        # 发布可视化
        if self.publish_viz and results.boxes is not None:
            viz_frame = results.plot()
            
            # 添加统计信息
            cv2.putText(
                viz_frame,
                f'Detections: {len(detections.detections)}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            try:
                viz_msg = self.bridge.cv2_to_imgmsg(viz_frame, encoding='bgr8')
                viz_msg.header = msg.header
                self.viz_pub.publish(viz_msg)
            except Exception as e:
                self.get_logger().error(f'可视化发布失败: {e}')
        
        # 定期打印统计
        if self.frame_count % 100 == 0:
            self.get_logger().info(
                f'已处理 {self.frame_count} 帧, '
                f'检测到 {self.detection_count} 个对象'
            )

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ObjectDetectorNode()
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
