import matplotlib.pyplot as plt
import numpy as np
import tkinter
import random
import Classifier_ML
from tkinter import filedialog
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
import data_processor as dp

FirstClassification_class = ['Normal', 'Abnormal']
SecondClassification_class = ['Normal', 'Blade Damage']











def ClassifyData(dir_path):
    # 데이터 로딩
    origin_images_index, origin_images = dl.LoadImageFromDir(dir_path)
    print("complete image loading")

    # 데이터 가공 ( 이진화 )
    images_binary_processed = dp.Processing_Binaryization(origin_images)
    print("complete image processing")

    # 데이터 정규화 ( 0~1사이로 변환 )
    images_binary_processed = images_binary_processed / 255
    print("start prediction!")

    # 1차 모델 로딩
    model_binary = load_model('model_predict/model_220811_dense4_Binaryization_processed_epoch3')
    model_binary.summary()

    # 1차 예측 ( 1차 분류를 위한 라벨(predictions 생성) )
    predictions1 = model_binary.predict(images_binary_processed)

    print("prediction count: " + str(len(predictions1)))

    # 1차 분류
    # data_group1 = np.empty((0, 510, 540), dtype=np.uint8)
    # data_group2 = np.empty((0, 510, 540), dtype=np.uint8)

    first_data_group = []
    first_data_group_index = []

    predict_partially_deposited_index_list = []
    predict_partially_deposited_prob_list = []

    predict_data_swelling = []
    predict_swelling_index_list = []
    predict_swelling_prob_list = []

    predict_data_bladeDamage = []
    predict_bladeDamage_index_list = []
    predict_bladeDamage_prob_list = []

    predict_data_overlapped = []
    predict_overlapped_index_list = []
    predict_overlapped_prob_list = []

    for i in range(len(origin_images)):
        # 예측값 정수화
        predicted_label = int(predictions1[i])
        if (predicted_label == 0):
            first_data_group.append(origin_images[i])
            first_data_group_index.append(origin_images_index[i])
        elif (predicted_label == 1):
            predict_partially_deposited_index_list.append(origin_images_index[i])
            predict_partially_deposited_prob_list.append(int(predictions1[i].mean() * 100))

    if (len(first_data_group) == 0):
        return predict_partially_deposited_index_list, \
               predict_partially_deposited_prob_list, \
               predict_swelling_index_list, \
               predict_swelling_prob_list, \
               predict_bladeDamage_index_list, \
               predict_bladeDamage_prob_list, \
               predict_overlapped_index_list, \
               predict_overlapped_prob_list

    first_data_group_np = np.array(first_data_group)
    # 1차 분류 테스트 출력
    print(first_data_group_np.shape)

    # 데이터 가공 ( 엣지 디텍션 )
    images_edge_processed = dp.Processing_EdgeDetection(first_data_group_np)

    # 데이터 정규화 ( 0~1사이로 변환 )
    images_edge_processed = images_edge_processed / 255

    # 2차 모델 로딩
    model_categorical = load_model('model_predict/model_221109_dense4_EdgeDetection_categorical_epoch6_batch16')
    model_categorical.summary()

    # 2차 예측 ( 2차 분류를 위한 라벨(predictions 생성) )
    second_prediction = model_categorical.predict(images_edge_processed)
    print(second_prediction.shape)


    for i in range(len(first_data_group_np)):
        predicted_label = np.argmax(second_prediction[i])
        if (predicted_label == 1):
            predict_data_swelling.append(first_data_group_np[i])
            predict_swelling_index_list.append(first_data_group_index[i])
            predict_swelling_prob_list.append(int(second_prediction[i][1] * 100))
        elif (predicted_label == 2):
            predict_data_bladeDamage.append(first_data_group_np[i])
            predict_bladeDamage_index_list.append(first_data_group_index[i])
            predict_bladeDamage_prob_list.append(int(second_prediction[i][2] * 100))
        elif (predicted_label == 3):
            predict_data_overlapped.append(first_data_group_np[i])
            predict_overlapped_index_list.append(first_data_group_index[i])
            predict_overlapped_prob_list.append(int(second_prediction[i][3] * 100))

    print(predict_partially_deposited_index_list)
    print(predict_swelling_index_list)
    print(predict_bladeDamage_index_list)
    print(predict_overlapped_index_list)

    return predict_partially_deposited_index_list, \
           predict_partially_deposited_prob_list, \
           predict_swelling_index_list, \
           predict_swelling_prob_list, \
           predict_bladeDamage_index_list, \
           predict_bladeDamage_prob_list, \
           predict_overlapped_index_list, \
           predict_overlapped_prob_list