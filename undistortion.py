#!/usr/bin/env python3
"""
图像畸变校正
展示径向畸变和切向畸变的校正效果
"""

import cv2
import numpy as np
import yaml

class UndistortionProcessor:
    def __init__(self, calib_file):
        """加载标定参数"""
        with open(calib_file, 'r') as f:
            data = yaml.safe_load(f)
        
        self.camera_matrix = np.array(data['camera_matrix'])
        self.dist_coeffs = np.array(data['distortion_coefficients'])
        self.img_size = (data['image_width'], data['image_height'])
        
        # 计算最优新相机矩阵
        # alpha=0: 裁剪掉所有无效像素(黑边)
        # alpha=1: 保留所有像素(包括黑边)
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeffs,
            self.img_size,
            alpha=0.5  # 折中方案
        )
        
        # 预计算映射表(加速实时处理)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.new_camera_matrix,
            self.img_size,
            cv2.CV_16SC2
        )
        
        print("畸变校正器已初始化")
        print(f"畸变系数: {self.dist_coeffs.ravel()}")
    
    def undistort(self, image, method='remap'):
        """
        校正图像畸变
        
        Args:
            image: 输入图像
            method: 'remap' (快) 或 'undistort' (慢)
        
        Returns:
            校正后的图像
        """
        if method == 'remap':
            # 使用预计算的映射表(推荐,速度快)
            dst = cv2.remap(image, self.map1, self.map2, 
                           cv2.INTER_LINEAR)
        else:
            # 直接校正(每次都重新计算,速度慢)
            dst = cv2.undistort(image, self.camera_matrix, 
                               self.dist_coeffs, None, 
                               self.new_camera_matrix)
        
        # 裁剪ROI
        x, y, w, h = self.roi
        if w > 0 and h > 0:
            dst = dst[y:y+h, x:x+w]
        
        return dst
    
    def visualize_distortion(self, image):
        """
        可视化畸变效果
        在图像上绘制网格,对比原图和校正后的网格
        """
        h, w = image.shape[:2]
        
        # 创建网格点
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, w-1, 10),
            np.linspace(0, h-1, 10)
        )
        points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        points = points.reshape(-1, 1, 2).astype(np.float32)
        
        # 原始网格
        img_grid = image.copy()
        for i in range(10):
            for j in range(10):
                pt = (int(grid_x[i, j]), int(grid_y[i, j]))
                cv2.circle(img_grid, pt, 3, (0, 255, 0), -1)
        
        # 校正后的网格
        img_corrected = self.undistort(image)
        
        # 将原始点映射到校正后的坐标
        points_undistorted = cv2.undistortPoints(
            points, 
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.new_camera_matrix
        )
        
        # 在校正图上绘制
        for pt in points_undistorted:
            x, y = int(pt[0, 0]), int(pt[0, 1])
            if 0 <= x < img_corrected.shape[1] and 0 <= y < img_corrected.shape[0]:
                cv2.circle(img_corrected, (x, y), 3, (0, 0, 255), -1)
        
        return img_grid, img_corrected

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='图像畸变校正')
    parser.add_argument('--calib', type=str, required=True,
                       help='标定文件路径')
    parser.add_argument('--camera', type=int, default=0,
                       help='相机ID')
    parser.add_argument('--mode', choices=['live', 'image'],
                       default='live', help='实时相机或单张图像')
    parser.add_argument('--input', type=str,
                       help='输入图像路径(image模式)')
    args = parser.parse_args()
    
    processor = UndistortionProcessor(args.calib)
    
    if args.mode == 'live':
        cap = cv2.VideoCapture(args.camera)
        
        print("按键说明:")
        print("  q - 退出")
        print("  g - 显示/隐藏网格")
        
        show_grid = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if show_grid:
                original, corrected = processor.visualize_distortion(frame)
                combined = np.vstack([original, corrected])
                cv2.imshow('Undistortion (Grid)', combined)
            else:
                corrected = processor.undistort(frame)
                combined = np.hstack([frame, corrected])
                
                cv2.putText(combined, "Original", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(combined, "Corrected",
                           (frame.shape[1] + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Undistortion', combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                show_grid = not show_grid
        
        cap.release()
        cv2.destroyAllWindows()
    
    else:  # image mode
        img = cv2.imread(args.input)
        if img is None:
            print(f"无法读取图像: {args.input}")
            return
        
        corrected = processor.undistort(img)
        
        # 显示对比
        combined = np.hstack([img, corrected])
        cv2.imshow('Undistortion Comparison', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
