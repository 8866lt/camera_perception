#!/usr/bin/env python3
"""
跌倒检测系统
基于躯干角度和髋部高度判断跌倒
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import argparse
import time

class FallDetector:
    def __init__(self, 
                 angle_threshold=60,
                 hip_threshold=0.8,
                 confidence_window=15,
                 confidence_ratio=0.7):
        """
        初始化跌倒检测器
        
        Args:
            angle_threshold: 躯干角度阈值(度),>此值判定为跌倒
            hip_threshold: 髋部高度阈值(归一化),>此值判定为低位
            confidence_window: 置信度窗口大小(帧数)
            confidence_ratio: 置信度比例,窗口内>此比例才报警
        """
        self.angle_threshold = angle_threshold
        self.hip_threshold = hip_threshold
        self.confidence_window = confidence_window
        self.confidence_ratio = confidence_ratio
        
        # 历史记录
        self.fall_history = deque(maxlen=confidence_window)
        
        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 报警状态
        self.alarm_active = False
        self.alarm_start_time = None
        
        print("跌倒检测器已初始化")
        print(f"  角度阈值: {angle_threshold}°")
        print(f"  髋部阈值: {hip_threshold}")
    
    def calculate_torso_angle(self, landmarks):
        """
        计算躯干角度(相对于垂直方向)
        
        Args:
            landmarks: MediaPipe关键点
            
        Returns:
            angle: 角度(度)
        """
        # 获取肩部和髋部中点
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        shoulder_mid = np.array([
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2
        ])
        
        hip_mid = np.array([
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2
        ])
        
        # 躯干向量
        torso_vector = hip_mid - shoulder_mid
        
        # 垂直向量(向下为正)
        vertical = np.array([0, 1])
        
        # 计算夹角
        cos_angle = np.dot(torso_vector, vertical) / (
            np.linalg.norm(torso_vector) * np.linalg.norm(vertical)
        )
        
        # 限制在[-1, 1]避免数值误差
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def get_hip_height(self, landmarks):
        """
        获取髋部高度(归一化坐标)
        
        Returns:
            height: 0-1之间,1表示图像底部
        """
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        hip_y = (left_hip.y + right_hip.y) / 2
        
        return hip_y
    
    def detect(self, frame):
        """
        检测是否跌倒
        
        Args:
            frame: 输入图像
            
        Returns:
            is_fall: 是否跌倒
            angle: 躯干角度
            hip_height: 髋部高度
            annotated_frame: 标注后的图像
        """
        # 转换颜色
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 姿态检测
        results = self.pose.process(rgb)
        
        annotated = frame.copy()
        is_fall = False
        angle = 0
        hip_height = 0
        
        if results.pose_landmarks:
            # 计算指标
            angle = self.calculate_torso_angle(results.pose_landmarks.landmark)
            hip_height = self.get_hip_height(results.pose_landmarks.landmark)
            
            # 判断是否跌倒
            is_horizontal = angle > self.angle_threshold
            is_low = hip_height > self.hip_threshold
            
            is_fall_current = is_horizontal and is_low
            
            # 添加到历史
            self.fall_history.append(1 if is_fall_current else 0)
            
            # 计算置信度
            if len(self.fall_history) >= self.confidence_window:
                fall_ratio = sum(self.fall_history) / len(self.fall_history)
                
                if fall_ratio > self.confidence_ratio:
                    is_fall = True
                    
                    # 触发报警
                    if not self.alarm_active:
                        self.alarm_active = True
                        self.alarm_start_time = time.time()
                else:
                    # 重置报警
                    if self.alarm_active and fall_ratio < 0.3:
                        self.alarm_active = False
            
            # 绘制骨架
            self.mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
            # 绘制指标
            h, w = frame.shape[:2]
            
            # 躯干角度
            color = (0, 0, 255) if angle > self.angle_threshold else (0, 255, 0)
            cv2.putText(annotated, f"Angle: {angle:.1f}deg", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # 髋部高度
            color = (0, 0, 255) if hip_height > self.hip_threshold else (0, 255, 0)
            cv2.putText(annotated, f"Hip Height: {hip_height:.2f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # 跌倒状态
            if is_fall:
                # 绘制大红框警告
                cv2.rectangle(annotated, (0, 0), (w, h), (0, 0, 255), 10)
                
                cv2.putText(annotated, "FALL DETECTED!", (w//2 - 200, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                
                # 显示报警时长
                if self.alarm_start_time:
                    alarm_duration = time.time() - self.alarm_start_time
                    cv2.putText(annotated, f"Duration: {alarm_duration:.1f}s",
                               (10, h - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 置信度
            if len(self.fall_history) > 0:
                confidence = sum(self.fall_history) / len(self.fall_history)
                cv2.putText(annotated, f"Confidence: {confidence:.2f}",
                           (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        else:
            cv2.putText(annotated, "No person detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return is_fall, angle, hip_height, annotated
    
    def release(self):
        """释放资源"""
        self.pose.close()

def main():
    parser = argparse.ArgumentParser(description='跌倒检测系统')
    parser.add_argument('--source', type=str, default='0',
                       help='输入源')
    parser.add_argument('--angle-threshold', type=float, default=60,
                       help='躯干角度阈值(度)')
    parser.add_argument('--hip-threshold', type=float, default=0.8,
                       help='髋部高度阈值')
    parser.add_argument('--alarm-sound', type=str, default=None,
                       help='报警声音文件')
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = FallDetector(
        angle_threshold=args.angle_threshold,
        hip_threshold=args.hip_threshold
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
    
    print("\n跌倒检测系统已启动")
    print("按 'q' 退出\n")
    
    # 报警音频(如果提供)
    alarm_player = None
    if args.alarm_sound:
        try:
            import pygame
            pygame.mixer.init()
            alarm_player = pygame.mixer.Sound(args.alarm_sound)
            print(f"已加载报警声音: {args.alarm_sound}")
        except ImportError:
            print("警告: pygame未安装,无法播放报警声音")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测跌倒
            is_fall, angle, hip_height, annotated = detector.detect(frame)
            
            # 播放报警声音
            if is_fall and alarm_player and detector.alarm_active:
                if not pygame.mixer.get_busy():
                    alarm_player.play()
            
            # 显示
            cv2.imshow('Fall Detection', annotated)
            
            # 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        
        if alarm_player:
            pygame.mixer.quit()

if __name__ == '__main__':
    main()
