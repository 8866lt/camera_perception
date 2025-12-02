#!/usr/bin/env python3
"""
RGB相机基础捕获示例
支持USB相机、RTSP流、CSI相机(Jetson)
"""

import cv2
import numpy as np
import argparse
from datetime import datetime

class CameraCapture:
    def __init__(self, source=0, width=640, height=480, fps=30):
        """
        初始化相机
        
        Args:
            source: 相机源
                    - 整数: USB相机ID (0, 1, 2...)
                    - 字符串: RTSP流 "rtsp://..."
                    - 字符串: CSI相机 "nvarguscamerasrc ! ..."
            width: 图像宽度
            height: 图像高度
            fps: 帧率
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        
        # 判断相机类型
        if isinstance(source, str) and 'nvarguscamerasrc' in source:
            # Jetson CSI相机
            gst_str = (
                f"nvarguscamerasrc ! "
                f"video/x-raw(memory:NVMM), width={width}, height={height}, "
                f"format=NV12, framerate={fps}/1 ! "
                f"nvvidconv flip-method=0 ! "
                f"video/x-raw, width={width}, height={height}, format=BGRx ! "
                f"videoconvert ! "
                f"video/x-raw, format=BGR ! appsink"
            )
            self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        else:
            # USB相机或RTSP流
            self.cap = cv2.VideoCapture(source)
            
            # 设置分辨率和帧率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开相机: {source}")
        
        # 实际获取到的参数(可能与设置值不同)
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"相机已打开:")
        print(f"  分辨率: {self.actual_width}x{self.actual_height}")
        print(f"  帧率: {self.actual_fps}")
        
        # 统计信息
        self.frame_count = 0
        self.start_time = datetime.now()
    
    def read(self):
        """读取一帧图像"""
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
        return ret, frame
    
    def get_fps(self):
        """计算实际运行帧率"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0
    
    def release(self):
        """释放相机"""
        self.cap.release()
        print(f"总帧数: {self.frame_count}")
        print(f"平均帧率: {self.get_fps():.2f} fps")

def main():
    parser = argparse.ArgumentParser(description='RGB相机捕获示例')
    parser.add_argument('--source', type=str, default='0',
                       help='相机源: USB相机ID或RTSP流地址')
    parser.add_argument('--width', type=int, default=640,
                       help='图像宽度')
    parser.add_argument('--height', type=int, default=480,
                       help='图像高度')
    parser.add_argument('--fps', type=int, default=30,
                       help='帧率')
    parser.add_argument('--save', action='store_true',
                       help='按s键保存图像')
    args = parser.parse_args()
    
    # 如果source是纯数字,转换为整数
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    # 创建相机对象
    camera = CameraCapture(source, args.width, args.height, args.fps)
    
    # 创建窗口
    cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
    
    print("\n操作说明:")
    print("  q - 退出")
    print("  s - 保存当前帧")
    print("  f - 显示/隐藏帧率")
    
    show_fps = True
    saved_count = 0
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("无法读取帧,退出...")
                break
            
            # 显示帧率
            if show_fps:
                fps_text = f"FPS: {camera.get_fps():.2f}"
                cv2.putText(frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示图像
            cv2.imshow('Camera', frame)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存图像
                filename = f"frame_{saved_count:04d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"已保存: {filename}")
                saved_count += 1
            elif key == ord('f'):
                show_fps = not show_fps
    
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
