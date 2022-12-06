import cv2
import numpy as np

# 创建浅绿色的背景
img = np.full((400, 400, 3), (0, 180, 0), dtype=np.uint8)

# 在图片中间画一个粉红色的圆
img = cv2.circle(img, (200, 200), 100, (255, 128, 255), thickness=-1)

# 保存图片
cv2.imwrite("image.png", img)
