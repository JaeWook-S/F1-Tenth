import numpy as np
import matplotlib.pyplot as plt
import yaml
import cv2
import os

# ===== 경로 =====
map_yaml = "./maps/Oschersleben.yaml"
waypoint_csv = "./maps/Oschersleben_centerline.csv"

# ===== yaml 읽기 =====
with open(map_yaml, 'r') as f:
    map_info = yaml.safe_load(f)

image_path = os.path.join(os.path.dirname(map_yaml), map_info['image'])
resolution = map_info['resolution']
origin = map_info['origin']  # [x, y, theta]

# ===== map 이미지 =====
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# ===== waypoint =====
waypoints = np.genfromtxt(waypoint_csv, delimiter=',')

# ===== world → pixel 변환 =====
def world_to_pixel(x, y):
    px = (x - origin[0]) / resolution
    py = (y - origin[1]) / resolution

    # 이미지 좌표계 뒤집힘 보정
    py = img.shape[0] - py

    return int(px), int(py)

# ===== waypoint 변환 =====
pixels = np.array([world_to_pixel(x, y) for x, y in waypoints[:, :2]])

# ===== 시각화 =====
plt.figure(figsize=(8, 8))
plt.imshow(img, cmap='gray')

plt.scatter(pixels[:, 0], pixels[:, 1], s=2, c='red')

plt.title("Map + Waypoints")
plt.axis('off')
plt.show()