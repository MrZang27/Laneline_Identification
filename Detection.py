import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def region_selection(edges):
    '''
    定义并剪切ROI区域
    '''
    
    # 定义ROI区域
    mask = np.zeros_like(edges)
    
    # 定义ROI区域的四个顶点
    if len(edges.shape) > 2:
        channel_count = edges.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    
    rows, cols = edges.shape[:2]

    #TODO: 修改调试ROI区域的四个顶点
    bottom_left = [cols * 0, rows * 0.8]
    top_left = [cols * 0.4, rows * 0.4]
    bottom_right = [cols * 0.9, rows * 0.9]
    top_right = [cols * 0.6, rows * 0.4]

    # 原ROI区域的四个顶点
    # bottom_left = [cols * 0.1, rows * 0.95]
    # top_left	 = [cols * 0.4, rows * 0.6]
    # bottom_right = [cols * 0.9, rows * 0.95]
    # top_right = [cols * 0.6, rows * 0.6]

    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    # 填充ROI区域
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # 返回ROI区域
    masked_edges = cv2.bitwise_and(edges, mask)

    if masked_edges is None:
        print("ROI区域选择失败")
    else:
        print("ROI区域选择成功")

    return masked_edges

def hough_transform(region):
    '''
    使用霍夫变换检测直线
    '''
    
    #TODO 定义霍夫变换的参数
    rho = 1
    theta = np.pi / 180
    threshold = 50 #增加阈值以减少干扰
    min_line_length = 20
    max_line_gap = 500
    
    # 使用霍夫变换检测直线
    lines = cv2.HoughLinesP(region, rho = rho, theta = theta, threshold = threshold,
						minLineLength = min_line_length, maxLineGap = max_line_gap)
    
    # 检查返回值
    if lines is None:
        print("在Hough变换中，没有检测到任何直线")
    else:
        print(f"在Hough变换中，检测到 {len(lines)} 条直线")
    
    return lines

