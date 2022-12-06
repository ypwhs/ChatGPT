import cv2
import numpy as np

# 读取图像
image = cv2.imread('test.png')

# 转换图像到 HSV 空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义黄色的颜色范围（可能需要根据图像的不同调整）
lower_yellow = (20, 100, 100)
upper_yellow = (30, 255, 255)

# 根据颜色范围提取黄色部分
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# 查找圆形
circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

# 获取查找到的圆形的坐标和半径
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    # 遍历每一个圆形
    for (x, y, r) in circles:
        # 在原图中画出圆形
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)

# 显示图像
cv2.imshow("Output", image)
cv2.waitKey(0)
