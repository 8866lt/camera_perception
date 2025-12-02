#!/usr/bin/env python3
"""
MediaPipe 基础姿态检测
支持单人实时姿态估计
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import time
from collections import deque

class PoseDetector:
    def __init__(self, 
                 static_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        初始化姿态检测器
        
        Args:
            static_mode: 静态图像模式(False=视频模式,启用跟踪)
            model_complexity: 模型复杂度 0=Lite, 1=Full, 2=Heavy
            smooth_landmarks: 是否平滑关键点
            enable_segmentation: 是否输出分割掩码
            min_detection_confidence: 检测置信度阈值
            min_tracking_confidence: 跟踪置信度阈值
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # 统计信息
        self.fps_counter = deque(maxlen=30)
        self.last_time = time.time()
        
        print(f"姿态检测器已初始化")
        print(f"  模型复杂度: {model_complexity}")
        print(f"  检测置信度: {min_detection_confidence}")
    
    def detect(self, image, draw=True):
        """
        检测姿态
        
        Args:
            image: BGR图像
            draw: 是否绘制关键点
            
        Returns:
            results: MediaPipe结果对象
            annotated_image: 标注后的图像
        """
        # BGR → RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 检测
        start = time.time()
        results = self.pose.process(rgb)
        inference_time = (time.time() - start) * 1000
        
        # 计算FPS
        current_time = time.time()
        self.fps_counter.append(1.0 / (current_time - self.last_time))
        self.last_time = current_time
        
        # 绘制
        annotated_image = image.copy()
        
        if results.pose_landmarks:
            if draw:
                # 绘制关键点和连接线
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            # 显示统计信息
            fps = np.mean(self.fps_counter)
            cv2.putText(annotated_image, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_image, f"Inference: {inference_time:.1f}ms", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(annotated_image, "No person detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return results, annotated_image
    
    def get_landmark_coords(self, results, landmark_id, image_shape):
        """
        获取关键点的像素坐标
        
        Args:
            results: MediaPipe结果
            landmark_id: 关键点ID
            image_shape: 图像尺寸(h, w)
            
        Returns:
            (x, y): 像素坐标,如果不存在返回None
        """
        if not results.pose_landmarks:
            return None
        
        landmark = results.pose_landmarks.landmark[landmark_id]
        h, w = image_shape[:2]
        
        return (int(landmark.x * w), int(landmark.y * h))
    
    def get_all_landmarks(self, results, image_shape=None):
        """
        获取所有关键点坐标
        
        Args:
            results: MediaPipe结果
            image_shape: 图像尺寸,如果提供则返回像素坐标
            
        Returns:
            landmarks: 关键点列表 [(x,y,z,visibility), ...]
        """
        if not results.pose_landmarks:
            return None
        
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            if image_shape is not None:
                h, w = image_shape[:2]
                x = landmark.x * w
                y = landmark.y * h
            else:
                x = landmark.x
                y = landmark.y
            
            landmarks.append({
                'x': x,
                'y': y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        
        return landmarks
    
    def release(self):
        """释放资源"""
        self.pose.close()

def main():
    parser = argparse.ArgumentParser(description='MediaPipe姿态检测')
    parser.add_argument('--source', type=str, default='0',
                       help='输入源:摄像头ID/视频文件/图片')
    parser.add_argument('--model', type=int, default=1, choices=[0, 1, 2],
                       help='模型复杂度: 0=Lite, 1=Full, 2=Heavy')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='检测置信度阈值')
    parser.add_argument('--output', type=str, default=None,
                       help='输出视频路径')
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = PoseDetector(
        model_complexity=args.model,
        min_detection_confidence=args.confidence
    )
    
    # 打开输入源
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"无法打开输入源: {source}")
        return
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n输入源信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps_in}")
    
    # 视频写入器
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps_in if fps_in > 0 else 30,
                                (width, height))
        print(f"将保存到: {args.output}")
    
    print("\n按键说明:")
    print("  q - 退出")
    print("  s - 保存当前帧")
    print("  SPACE - 暂停/继续")
    
    paused = False
    frame_count = 0
    
    try:
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("视频结束")
                    break
                
                frame_count += 1
                
                # 检测姿态
                results, annotated = detector.detect(frame, draw=True)
                
                # 显示
                cv2.imshow('MediaPipe Pose', annotated)
                
                # 保存视频
                if writer is not None:
                    writer.write(annotated)
            else:
                cv2.imshow('MediaPipe Pose', annotated)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"pose_frame_{frame_count:06d}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"已保存: {filename}")
            elif key == ord(' '):
                paused = not paused
                print("暂停" if paused else "继续")
    
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        detector.release()
        
        print(f"\n总计处理 {frame_count} 帧")

if __name__ == '__main__':
    main()
