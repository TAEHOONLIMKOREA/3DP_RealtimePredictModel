import grpc
import RealTimeModel_Serving_pb2
import RealTimeModel_Serving_pb2_grpc
from concurrent import futures
import Classifier_ML
import cv2
import numpy as np



class RealTimeDetectServicer(RealTimeModel_Serving_pb2_grpc.RealTimeModelServingServicer):
    def DefectDetection(self, request, context):
        # 작동 메세지 출력
        print("Check Connection")

        # 현재는 dbName에 디비이름 대신 사진파일 저장경로가 들어가 있음
        data = request.Datas
        encoded_img = np.fromstring(data, dtype=np.uint8)
        img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)


        # 디비명을 가지고 이미지 다운로드
        print("complete download images")
        # 다운로드 후 저장된 사진들의 디렉토리 경로를 전달

        result_class, result_percent = Classifier_ML.ClassifyData(img)

        # After Classification, deliver to grpc_result object
        predict_result = RealTimeModel_Serving_pb2.PredictResult()
        predict_result.ClassResult = result_class
        predict_result.ClassReusltPercent = result_percent

        return predict_result




def StartServer() -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    RealTimeModel_Serving_pb2_grpc.add_RealTimeModelServingServicer_to_server(RealTimeDetectServicer(), server)
    print("server started!")
    server.add_insecure_port('[::]:50052')
    server.start()
    # server.wait_for_termination()
    while (1):
        user_input = input()
        if (user_input == "stop"):
            server.stop(0)
            return
