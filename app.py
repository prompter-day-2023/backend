from flask import Flask
from flask import request
import cv2
import numpy as np
from datetime import datetime

app = Flask(__name__)

@app.route('/', methods=['POST'])
def hello():
    file = request.files['file']
    file_name = file.filename.split('.')[0]
    image_type = file.filename.split('.')[-1]
    created_at = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    new_file_name = f"{file_name}-{created_at}.{image_type}"

    # 파일 데이터를 읽어와 NumPy 배열로 변환
    file_data = file.read()
    np_array = np.frombuffer(file_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray_img, 100, 250)

    # 엣지 확장을 위한 커널 생성
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=1)

    # 배경색과 엣지색을 지정
    background_color = (255, 255, 255)  # 흰색 배경
    edge_color = (0, 0, 0)  # 검은색 엣지

    # 배경 부분을 원하는 배경색으로 채우기
    filled_image = np.full_like(image, background_color)
    filled_image[dilated != 0] = edge_color

    cv2.imwrite(new_file_name, filled_image)

    return { "status": 200, "message": 'OK' }

if __name__ == '__main__':
    app.run(debug=True, port=5000)