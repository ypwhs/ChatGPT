import cv2
import numpy as np

# 读取图像
image = cv2.imread("my_image.jpg")

# 转换为HSV颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义黄色的HSV颜色范围
lower_yellow = (20, 100, 100)
upper_yellow = (30, 255, 255)

# 根据颜色范围创建掩模
mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

# 找到图像中的圆形
circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

# 如果找到圆形，绘制它们
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    for (x, y, r) in circles:
        # 绘制圆形
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)
        # 绘制圆心
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

# 显示结果图像
cv2.imshow("Image", image)
cv2.waitKey(0)

# 保存图像到本地
cv2.imwrite("detected_circles.jpg", image)
