import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def average_slope_intercept(lines):
    '''
    计算直线的斜率和截距
    '''
    left_lines = [] #(slope, intercept)
    left_weights = [] #(length,)
    right_lines = [] #(slope, intercept)
    right_weights = [] #(length,)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1==x2:
                continue

            # 计算直线的斜率和截距    
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)

            # 区分左右车道线
            # 左车道线的斜率为负
            # 右车道线的斜率为正
            # 过于接近水平的线条将被忽略
            
            #TODO: 修改斜率的阈值
            if abs(slope) < 0.2:
                continue

            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
            
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane

def pixel_points(y1, y2, line):
    '''
    计算直线的两个端点
    '''

    if line is None:
        return None
    
    slope, intercept = line

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)

    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    '''
    从pixel_points函数获取直线的两个端点并创建车道线
    '''
    left_lane, right_lane = average_slope_intercept(lines)

    y1 = image.shape[0]
    y2 = y1 * 0.6

    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)

    return left_line, right_line

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=20):
    '''
    绘制车道线
    '''
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)