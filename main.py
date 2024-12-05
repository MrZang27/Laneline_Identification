import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from Detection import region_selection,hough_transform
from DrawLaneLine import lane_lines,draw_lane_lines

def frame_processor(image):
    '''
    将输入的图像转换为灰度图像，并进行高斯模糊处理
    '''
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯模糊处理
    kernel_size = 5
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # 边缘检测
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur, low_threshold, high_threshold)
    # cv2.imshow('edges',edges)
    # cv2.waitKey(0)

    # 选择ROI区域
    region = region_selection(edges)
    #print(type(region))
    # cv2.imshow('region',region)
    # cv2.waitKey(0)

    # 霍夫变换检测直线
    hough = hough_transform(region)
    #print(type(hough))

    # 绘制车道线
    result = draw_lane_lines(image,lane_lines(image,hough))

    return result

if __name__ =='__main__':
    image_path = './images/'

    result_path = './result_images/'

    # 创建一个目录用于保存结果
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    # 获取目录下的所有文件和文件夹
    image_names = os.listdir(image_path) #list

    # 创建一个窗口
    cv2.namedWindow('Lane Detection', cv2.WINDOW_NORMAL)

    for image_name in image_names:
      
        # 读取图像
        image = cv2.imread(image_path + image_name)
        print(image.shape[:2])
        
        # 处理图像
        result = frame_processor(image)
        
        # 显示图像
        cv2.imshow('Lane Detection', result)

        # 保存图像
        # cv2.imwrite(f'./result_images/{image_name}_result.jpg', result)
        
        # 等待按键
        cv2.waitKey(0)

    # 关闭窗口
    cv2.destroyAllWindows()


