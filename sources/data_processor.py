import cv2
import numpy as np


# First Procsses
def Processing_Binaryization(input_image):
    thresh = 230
    maxValue = 255

    # 이미지 이진화 - thresholding: 0~230 사이의 값들을 전부 0으로 바꿔줌.. 230~255 값들을 전부 255로 바꿔줌
    th, bin_img = cv2.threshold(input_image, thresh, maxValue, cv2.THRESH_BINARY)

    return bin_img


# Second Procsses
def Processing_EdgeDetection(input_image):

    # 흑백 반전
    reverse_out = 255 - input_image

    # 블러링
    blur = cv2.GaussianBlur(reverse_out, (9, 9), 0)

    # 엣지 디텍션
    edge_out = cv2.Canny(blur, 0, 25)

    # 커널 생성
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 오프닝
    dst_dilate = cv2.dilate(edge_out, k)

    return dst_dilate