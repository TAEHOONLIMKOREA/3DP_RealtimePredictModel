import cv2
import numpy as np


# First Procsses
def Processing_Binaryization(input_images):
    thresh = 230
    maxValue = 255
    proceessed_images = []
    for i in range(len(input_images)):
        # 이미지 이진화 - thresholding: 0~230 사이의 값들을 전부 0으로 바꿔줌.. 230~255 값들을 전부 255로 바꿔줌
        th, bin_img = cv2.threshold(input_images[i], thresh, maxValue, cv2.THRESH_BINARY)
        proceessed_images.append(bin_img)

    proceessed_images_np = np.array(proceessed_images)
    return proceessed_images_np


# Second Procsses
def Processing_EdgeDetection(input_images):

    proceessed_images = []
    for i in range(len(input_images)):
        # 흑백 반전
        reverse_out = 255 - input_images[i]

        # 블러링
        blur = cv2.GaussianBlur(reverse_out, (9, 9), 0)

        # 엣지 디텍션
        edge_out = cv2.Canny(blur, 0, 25)

        # 커널 생성
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # 오프닝
        dst_dilate = cv2.dilate(edge_out, k)
        proceessed_images.append(dst_dilate)

    proceessed_images_np = np.array(proceessed_images)
    return proceessed_images_np