import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from depth_anything_v2.dpt import DepthAnythingV2

# 1. 配置设备 (MacBook M系列芯片使用 mps)
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def get_model():
    # 使用 vitb (Small) 以保证在 MacBook 上的运行速度
    model_configs = {'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}}
    model = DepthAnythingV2(**model_configs['vitb'])
    # 请确保路径下有该权重文件
    model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vitb.pth', map_location='cpu'))
    return model.to(DEVICE).eval()

def detect_obstacles(raw_img, depth_map, sensitivity=0.15, area_threshold=0.03):
    """
    sensitivity: 灵敏度。数值越小，越容易把稍微凸出的东西当成障碍物。
    area_threshold: 面积阈值。占画面百分之几才触发输出 1。
    """
    # DA2 输出的深度图，数值越大越近。
    # 归一化到 0-1
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    # 使用直方图众数找到货架主体深度
    # 我们取 0.1 到 0.9 之间的分布，排除极远和极近的干扰
    hist, bins = np.histogram(depth_norm, bins=100, range=(0.1, 0.9))
    shelf_depth_bin = bins[np.argmax(hist)] 
    
    # 判定逻辑：当前深度 > 货架主体深度 + 偏移量
    # 注意：DA2 越近值越大，所以障碍物是比货架深度大的部分
    obstacle_mask = (depth_norm > (shelf_depth_bin + sensitivity)).astype(np.uint8)
    
    # 形态学开运算：去除孤立的小噪点
    kernel = np.ones((5, 5), np.uint8)
    obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel)
    
    # 计算障碍物面积占比
    occupancy = np.sum(obstacle_mask) / obstacle_mask.size
    is_obstacle_present = 1 if occupancy > area_threshold else 0
    
    return is_obstacle_present, obstacle_mask, occupancy

def visualize_result(raw_img, mask, is_present, score):
    # 将 mask 转换为红色遮罩叠加在原图上
    color_mask = np.zeros_like(raw_img)
    color_mask[mask == 1] = [0, 0, 255] # BGR 格式的红色
    
    # 融合原图和遮罩 (权重 0.7 原图, 0.3 红色)
    overlay = cv2.addWeighted(raw_img, 0.7, color_mask, 0.3, 0)
    
    # 在图上写字
    label = f"Obstacle: {'YES' if is_present else 'NO'} ({score:.1%})"
    color = (0, 0, 255) if is_present else (0, 255, 0)
    cv2.putText(overlay, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return overlay

def visualize_logic(depth_map, sensitivity):
    # 归一化深度用于绘图
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    # 重新计算直方图用于显示
    hist, bins = np.histogram(depth_norm, bins=100, range=(0.1, 0.9))
    shelf_depth_bin = bins[np.argmax(hist)]
    threshold_line = shelf_depth_bin + sensitivity

    plt.figure(figsize=(10, 4))
    plt.bar(bins[:-1], hist, width=0.008, color='gray', alpha=0.6, label='Depth Distribution')
    
    # 画出系统判定的货架位置
    plt.axvline(shelf_depth_bin, color='blue', linestyle='--', label=f'Shelf Base (Mode): {shelf_depth_bin:.2f}')
    
    # 画出判定障碍物的阈值线
    plt.axvline(threshold_line, color='red', label=f'Obstacle Threshold: {threshold_line:.2f}')
    
    plt.title("Statistical Depth Analysis (How we find the Shelf)")
    plt.xlabel("Relative Depth (Larger = Closer)")
    plt.ylabel("Pixel Count")
    plt.legend()
    plt.show()



# --- 主程序 ---
if __name__ == "__main__":
    model = get_model()
    
    # 替换为你测试图片的路径
    image_path = '/Users/chenyi/Developer/Depth and obsacals/2025-12-17-17-56-sc5x_000017D8-0e79e296-c0c7-4d7a-b1cf-3a887d65118e.jpg' 
    raw_img = cv2.imread(image_path)
    
    if raw_img is None:
        print("Error: Could not read image.")
    else:
        # 推理深度图
        depth = model.infer_image(raw_img)
        
        # 识别障碍物
        has_obs, obs_mask, ratio = detect_obstacles(raw_img, depth)
        
        # 输出布尔值给后续相机
        print(f"RESULT: {has_obs}")

        depth = model.infer_image(raw_img)
        visualize_logic(depth, 0.15)
        
        # 可视化
        vis_img = visualize_result(raw_img, obs_mask, has_obs, ratio)
        
        # 显示结果
        cv2.imshow('Obstacle Detection Pipeline', vis_img)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()