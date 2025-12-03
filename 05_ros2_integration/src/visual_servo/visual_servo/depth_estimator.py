#!/usr/bin/env python3
"""
深度估计节点
将2D检测框转换为3D位姿
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PoseStamped, Point
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs
import numpy as np

class DepthEstimatorNode(Node):
    def __init__(self):
        super().__init__('depth_estimator')
        
        # 声明参数
        self.declare_parameter('depth_topic', '/camera/depth')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_optical_frame')
        
        # 获取参数
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.base_frame = self.get_parameter('base_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        
        # 初始化
        self.bridge = CvBridge()
        self.camera_info = None
        self.depth_image = None
        self.detections = None
        
        # tf2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # 订阅
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10
        )
        
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )
        
        # 发布
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/object_pose_3d',
            10
        )
        
        self.get_logger().info('深度估计节点已启动')
    
    def camera_info_callback(self, msg):
        """相机参数回调"""
        if self.camera_info is None:
            self.camera_info = msg
            K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(f'已接收相机参数\n{K}')
    
    def depth_callback(self, msg):
        """深度图回调"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='passthrough'
            )
        except Exception as e:
            self.get_logger().error(f'深度图转换失败: {e}')
    
    def detection_callback(self, msg):
        """检测结果回调"""
        self.detections = msg
        self._process_detections()
    
    def _process_detections(self):
        """处理检测结果"""
        if (self.detections is None or
            self.depth_image is None or
            self.camera_info is None):
            return
        
        if len(self.detections.detections) == 0:
            return
        
        # 处理第一个检测
        detection = self.detections.detections[0]
        
        # 边界框中心
        cx = int(detection.bbox.center.position.x)
        cy = int(detection.bbox.center.position.y)
        
        # 获取深度值
        depth = self.depth_image[cy, cx]
        
        if depth == 0 or np.isnan(depth):
            self.get_logger().warn('深度值无效')
            return
        
        # 转换为米
        depth_m = float(depth) / 1000.0
        
        # 相机内参
        K = np.array(self.camera_info.k).reshape(3, 3)
        fx = K[0, 0]
        fy = K[1, 1]
        cx_cam = K[0, 2]
        cy_cam = K[1, 2]
        
        # 反投影到3D
        X = (cx - cx_cam) * depth_m / fx
        Y = (cy - cy_cam) * depth_m / fy
        Z = depth_m
        
        # 创建位姿消息(相机坐标系)
        pose_camera = PoseStamped()
        pose_camera.header = self.detections.header
        pose_camera.header.frame_id = self.camera_frame
        pose_camera.pose.position.x = X
        pose_camera.pose.position.y = Y
        pose_camera.pose.position.z = Z
        pose_camera.pose.orientation.w = 1.0
        
        # 坐标变换到基座
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rclpy.time.Time()
            )
            
            pose_base = tf2_geometry_msgs.do_transform_pose(
                pose_camera,
                transform
            )
            
            # 发布
            self.pose_pub.publish(pose_base)
            
            self.get_logger().info(
                f'目标位置(基座): '
                f'x={pose_base.pose.position.x:.3f}, '
                f'y={pose_base.pose.position.y:.3f}, '
                f'z={pose_base.pose.position.z:.3f}'
            )
        
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f'坐标变换失败: {ex}')

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = DepthEstimatorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
