import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ 코드 이후에 tensorflow를 import 할 것.
import sys
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.models import load_model
from sources import data_processor as dp
import cv2


def ClassifyData(img):
    # 모델에 들어가는 사이즈로 변경
    img = cv2.resize(img, dsize=(540, 510), interpolation=cv2.INTER_AREA)
    # 데이터 로딩
    print("complete image loading")

    # 데이터 가공 ( 이진화 )
    image_binary_processed = dp.Processing_Binaryization(img)
    print("complete image processing")

    # 데이터 정규화 ( 0~1사이로 변환 )
    image_binary_processed = image_binary_processed / 255
    print("start prediction!")

    # Model의 Input이 3차원 배열(이미지들의 집합)이기 때문에 한장을 분류하더라도 배열로 만들어줘야함.
    input_img_array = np.array([image_binary_processed])

    # 1차 모델 로딩
    model_binary = load_model('ClassificationModels/model_220811_dense4_Binaryization_processed_epoch3')
    # model_binary.summary()

    #     # 1차 예측 ( 1차 분류를 위한 라벨(predictions 생성) )
    first_predictions = model_binary.predict(input_img_array)

    print("prediction count: " + str(len(first_predictions)))

    # 1차 분류

    # 예측값 정수화
    predicted_label = np.argmax(first_predictions[0])
    global result_class
    global result_percent

    if (predicted_label == 0):
        # 데이터 가공 ( 엣지 디텍션 )
        img_edge_processed = dp.Processing_EdgeDetection(img)

        # 데이터 정규화 ( 0~1사이로 변환 )
        img_edge_processed = img_edge_processed / 255

        # Model의 Input규격이 3차원 배열(이미지들의 집합)이기 때문에 한장을 분류하더라도 배열로 만들어줘야함.
        input_img_array2 = np.array([img_edge_processed])

        # 2차 모델 로딩
        model_categorical = load_model('ClassificationModels/model_221109_dense4_EdgeDetection_categorical_epoch6_batch16')
        # model_categorical.summary()

        # 2차 예측 ( 2차 분류를 위한 라벨(predictions 생성) )
        second_prediction = model_categorical.predict(input_img_array2)
        print(second_prediction.shape)
        print(second_prediction)

        predicted_label = np.argmax(second_prediction[0])
        if (predicted_label == 0):
            result_class = "Normal"
            result_percent = int(second_prediction[0][0] * 100)
        elif (predicted_label == 1):
            result_class = "Swelling"
            result_percent = int(second_prediction[0][1] * 100)
        elif (predicted_label == 2):
            result_class = "BladeDamage"
            result_percent = int(second_prediction[0][2] * 100)
        elif (predicted_label == 3):
            result_class = "Overlapped"
            result_percent = int(second_prediction[0][3] * 100)

    # 첫 번째 분류에서 라벨이 1로 나온 경우
    elif (predicted_label == 1):
        result_class = "PartiallyDeposited"
        result_percent = int(first_predictions[0].mean() * 100)


    return result_class, result_percent


