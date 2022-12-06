import cv2
import numpy as np

# 读取图片
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# 计算大于 100 灰阶的均值
mean = np.mean(image[image > 100])

# 将图片的灰阶范围控制在均值正负 20 的范围内
image[image > mean + 20] = mean + 20
image[image < mean - 20] = mean - 20

# 保存处理后的图片
cv2.imwrite('image_processed.png', image)

