from flask import Flask
import cv2
import numpy as np
from flask import jsonify, request

# app = Flask(__name__)

# @app.route('/')
def hello():
    # file = request.files['file']
    image = cv2.imread('./aaa.jpeg')

    gray_img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray_img, 100, 250)

    # 배경색과 엣지색을 지정
    background_color = (255, 255, 255)  # 흰색 배경
    edge_color = (0, 0, 0)  # 검은색 엣지

    # 배경 부분을 원하는 배경색으로 채우기
    filled_image = np.full_like(image, background_color)
    filled_image[edged != 0] = edge_color

    # cv2.imshow(filled_image)

    # 엣지 확장을 위한 커널 생성
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=1)

    # 배경색과 엣지색을 지정
    background_color = (255, 255, 255)  # 흰색 배경
    edge_color = (0, 0, 0)  # 검은색 엣지

    # 배경 부분을 원하는 배경색으로 채우기
    filled_image = np.full_like(image, background_color)
    filled_image[dilated != 0] = edge_color

    # cv2.imshow(filled_image)
    cv2.imwrite('result.png', filled_image)

hello()