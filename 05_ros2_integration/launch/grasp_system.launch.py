"""
完整抓取系统Launch文件
启动所有必要的节点
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 获取包路径
    camera_share = get_package_share_directory('camera_publisher')
    detector_share = get_package_share_directory('object_detector')
    servo_share = get_package_share_directory('visual_servo')
    
    # 相机配置
    camera_config = os.path.join(camera_share, 'config', 'camera_params.yaml')
    detector_config = os.path.join(detector_share, 'config', 'detector_params.yaml')
    servo_config = os.path.join(servo_share, 'config', 'servo_params.yaml')
    
    # 1. 相机节点
    camera_node = Node(
        package='camera_publisher',
        executable='camera_node',
        name='camera_publisher',
        output='screen',
        parameters=[camera_config]
    )
    
    # 2. 检测节点
    detector_node = Node(
        package='object_detector',
        executable='detector_node',
        name='object_detector',
        output='screen',
        parameters=[detector_config]
    )
    
    # 3. 深度估计节点
    depth_estimator = Node(
        package='visual_servo',
        executable='depth_estimator',
        name='depth_estimator',
        output='screen',
        parameters=[servo_config]
    )
    
    # 4. 视觉伺服节点
    servo_node = Node(
        package='visual_servo',
        executable='servo_node',
        name='visual_servo',
        output='screen',
        parameters=[servo_config]
    )
    
    # 5. 静态tf发布(相机到基座的变换)
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_tf_publisher',
        arguments=[
            '0.1', '0', '0.5',  # x, y, z (米)
            '0', '0', '0', '1',  # qx, qy, qz, qw
            'base_link',
            'camera_optical_frame'
        ]
    )
    
    # 6. RViz可视化(可选)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', os.path.join(servo_share, 'config', 'grasp.rviz')],
        condition=lambda context: False  # 默认不启动,需要时改为True
    )
    
    return LaunchDescription([
        camera_node,
        detector_node,
        depth_estimator,
        servo_node,
        static_tf,
        # rviz_node  # 取消注释以启动RViz
    ])
