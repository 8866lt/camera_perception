#!/usr/bin/env python3
"""
深蹲姿态评估系统
评估深蹲动作是否标准,给出评分和改进建议
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
from collections import deque

class SquatEvaluator:
    def __init__(self):
        """初始化深蹲评估器"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 标准参数
        self.standards = {
            'knee_angle_min': 70,    # 膝盖最小角度
            'knee_angle_max': 100,   # 膝盖最大角度
            'back_angle_max': 20,    # 背部与垂直线夹角
            'hip_knee_angle_min': 80  # 髋-膝-踝角度
        }
        
        # 计数器
        self.squat_count = 0
        self.state = 'up'  # 'up' or 'down'
        self.knee_angle_history = deque(maxlen=10)
        
        print("深蹲评估器已初始化")
        print(f"标准:")
        for key, value in self.standards.items():
            print(f"  {key}: {value}")
    
    def calculate_angle(self, a, b, c):
        """计算三点夹角"""
        a_pos = np.array([a.x, a.y])
        b_pos = np.array([b.x, b.y])
        c_pos = np.array([c.x, c.y])
        
        ba = a_pos - b_pos
        bc = c_pos - b_pos
        
        cos_angle = np.dot(ba, bc) / (
            np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def evaluate(self, landmarks, image_shape):
        """
        评估深蹲姿态
        
        Args:
            landmarks: MediaPipe关键点
            image_shape: 图像尺寸
            
        Returns:
            evaluation: 评估结果字典
        """
        # 获取关键点
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_foot = landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
        
        # 1. 膝盖角度(髋-膝-踝)
        knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        self.knee_angle_history.append(knee_angle)
        
        # 2. 背部角度(相对于垂直方向)
        back_angle = self._calculate_back_angle(left_shoulder, left_hip)
        
        # 3. 检查膝盖是否超过脚尖
        knee_over_toe = left_knee.x > left_foot.x
        
        # 4. 检查膝盖内扣
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        knee_valgus = self._check_knee_valgus(
            left_knee, right_knee, left_ankle,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        )
        
        # 评分
        score = 100
        issues = []
        
        # 膝盖角度
        if knee_angle < self.standards['knee_angle_min']:
            issues.append("膝盖弯曲过度")
            score -= 20
        elif knee_angle > self.standards['knee_angle_max']:
            issues.append("蹲得不够深")
            score -= 15
        
        # 背部角度
        if back_angle > self.standards['back_angle_max']:
            issues.append("背部过度前倾")
            score -= 25
        
        # 膝盖超过脚尖
        if knee_over_toe:
            issues.append("膝盖超过脚尖")
            score -= 30
        
        # 膝盖内扣
        if knee_valgus:
            issues.append("膝盖内扣")
            score -= 20
        
        score = max(0, score)
        
        # 更新计数
        self._update_count(knee_angle)
        
        return {
            'score': score,
            'knee_angle': knee_angle,
            'back_angle': back_angle,
            'knee_over_toe': knee_over_toe,
            'knee_valgus': knee_valgus,
            'issues': issues,
            'count': self.squat_count,
            'state': self.state
        }
    
    def _calculate_back_angle(self, shoulder, hip):
        """计算背部角度"""
        shoulder_pos = np.array([shoulder.x, shoulder.y])
        hip_pos = np.array([hip.x, hip.y])
        
        back_vector = shoulder_pos - hip_pos
        vertical = np.array([0, -1])  # 垂直向上
        
        cos_angle = np.dot(back_vector, vertical) / (
            np.linalg.norm(back_vector) + 1e-6
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def _check_knee_valgus(self, left_knee, right_knee, left_ankle, right_ankle):
        """检查膝盖内扣"""
        # 计算膝盖间距和脚踝间距
        knee_dist = abs(left_knee.x - right_knee.x)
        ankle_dist = abs(left_ankle.x - right_ankle.x)
        
        # 如果膝盖间距小于脚踝间距,说明膝盖内扣
        return knee_dist < ankle_dist * 0.8
    
    def _update_count(self, knee_angle):
        """更新深蹲计数"""
        # 状态机
        if self.state == 'up' and knee_angle < 100:
            self.state = 'down'
        elif self.state == 'down' and knee_angle > 140:
            self.state = 'up'
            self.squat_count += 1
    
    def visualize(self, frame, landmarks, evaluation):
        """
        可视化评估结果
        
        Args:
            frame: 输入图像
            landmarks: MediaPipe关键点
            evaluation: 评估结果
            
        Returns:
            annotated: 标注后的图像
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # 绘制骨架
        self.mp_drawing.draw_landmarks(
            annotated,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )
        
        # 显示评分
        score = evaluation['score']
        color = self._get_score_color(score)
        
        cv2.putText(annotated, f"Score: {score}/100",
                   (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # 显示计数
        cv2.putText(annotated, f"Count: {evaluation['count']}",
                   (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        
        # 显示角度
        cv2.putText(annotated, f"Knee: {evaluation['knee_angle']:.1f}deg",
                   (10, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        cv2.putText(annotated, f"Back: {evaluation['back_angle']:.1f}deg",
                   (10, 175),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # 显示问题
        y = h - 150
        if evaluation['issues']:
            cv2.putText(annotated, "Issues:",
                       (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y += 35
            
            for issue in evaluation['issues']:
                cv2.putText(annotated, f"  - {issue}",
                           (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y += 30
        else:
            cv2.putText(annotated, "Perfect Form!",
                       (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 绘制角度弧线(膝盖)
        self._draw_angle_arc(annotated, landmarks, evaluation['knee_angle'])
        
        return annotated
    
    def _get_score_color(self, score):
        """根据评分返回颜色"""
        if score >= 90:
            return (0, 255, 0)  # 绿色
        elif score >= 70:
            return (0, 255, 255)  # 黄色
        else:
            return (0, 0, 255)  # 红色
    
    def _draw_angle_arc(self, image, landmarks, angle):
        """绘制膝盖角度弧线"""
        h, w = image.shape[:2]
        
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        
        # 转换为像素坐标
        knee_x = int(left_knee.x * w)
        knee_y = int(left_knee.y * h)
        
        # 绘制圆弧
        radius = 50
        color = self._get_score_color(100 - abs(angle - 90))
        
        cv2.ellipse(image, (knee_x, knee_y), (radius, radius),
                   0, 0, int(angle), color, 2)
    
    def detect_and_evaluate(self, frame):
        """
        检测姿态并评估
        
        Args:
            frame: 输入图像
            
        Returns:
            evaluation: 评估结果
            annotated: 标注后的图像
        """
        # 转换颜色
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 姿态检测
        results = self.pose.process(rgb)
        
        evaluation = None
        annotated = frame.copy()
        
        if results.pose_landmarks:
            # 评估
            evaluation = self.evaluate(
                results.pose_landmarks.landmark,
                frame.shape
            )
            
            # 可视化
            annotated = self.visualize(
                frame,
                results.pose_landmarks,
                evaluation
            )
        else:
            cv2.putText(annotated, "No person detected", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return evaluation, annotated
    
    def release(self):
        """释放资源"""
        self.pose.close()

def main():
    parser = argparse.ArgumentParser(description='深蹲姿态评估系统')
    parser.add_argument('--source', type=str, default='0',
                       help='输入源')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = SquatEvaluator()
    
    # 打开输入源
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"无法打开输入源: {source}")
        return
    
    print("\n深蹲姿态评估系统已启动")
    print("开始深蹲,系统将实时评估您的姿态")
    print("\n按 'q' 退出")
    print("按 'r' 重置计数\n")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 评估
            evaluation, annotated = evaluator.detect_and_evaluate(frame)
            
            # 显示
            cv2.imshow('Squat Evaluation', annotated)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                evaluator.squat_count = 0
                print("计数已重置")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        evaluator.release()
        
        print(f"\n训练统计:")
        print(f"  总计完成: {evaluator.squat_count} 个深蹲")

if __name__ == '__main__':
    main()
