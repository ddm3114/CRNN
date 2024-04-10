import numpy as np
import matplotlib.pyplot as plt
import cv2

# 展示灰度图
def image_show(image, title = "", gray=True):
    if(gray):
        plt.set_cmap('gray')
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    path = '5.png'  #'dataset/cropped_lps/cropped_lps/10.jpg'
    image = cv2.imread(path)                            # 读取图片
    #image = cv2.resize(image, (500,160), interpolation=cv2.INTER_LINEAR)
    original_image = image.copy()
    image_show(image, 'original')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     # 转化为灰度图
    image_show(image, 'gray')

    image = cv2.bilateralFilter(image, 3, 75, 75)       # 平滑化
    image_show(image, 'smooth')

    image = cv2.Canny(image, 70, 400)                   # 提取边界
    image_show(image, 'edge')

    contours, new = cv2.findContours(image.copy(),      # 寻找轮廓(面积最大的5个)
                cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    copy = original_image.copy()
    _ = cv2.drawContours(copy, contours, -1, (255, 0, 0), 2)
    image_show(copy, 'contour', gray=False)

    plate = None
    count = 0

    # 检测不同轮廓图的信息，寻找最符合的（最接近车牌的）
    for i in contours:
        a = cv2.arcLength(i, True)
        edge_count = cv2.approxPolyDP(i, 0.02 * a, True)
        x, y, w, h = cv2.boundingRect(i)
        if(h>10  and w/h > 1.5 and w/h < 6 
           and len(edge_count) > 3 and len(edge_count) < 7):
            count += 1
            plate = original_image[y:y+h, x:x+w]
            image_show(plate, 'plate' + str(count)+' ('+ str(len(edge_count))+')', gray=False)

