from flask import Flask
from flask import request, jsonify
import cv2
import numpy as np
import openai
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

openai.api_key = os.getenv('GPT_API_KEY')

@app.route('/diary', methods=['POST'])
def createDiary():
    contents = request.json.get('contents')

    # TODO: GPT 요약 프롬프트 연결하기
    response = openai.Completion.create(
        model = 'text-davinci-003',   # openai에서 제공하는 모델 입력 (GPT-3.5)
        prompt = '오늘은 친구들과 떡볶이를 먹었어',  # 원하는 실행어 입력
        temperature = 0,
        max_tokens = 100,   # 입력 + 출력 값으로 잡을 수 있는 max_tokens 값
        top_p = 1,
        frequency_penalty = 0.0,
        presence_penalty = 0.0,
        stop = ['\n']   # stop 지점 설정
    )
    # TODO: DALLE 연결하기

    print(response);

    return {'response': response.choices[0].text.strip()}


@app.route('/result', methods=['POST'])
def createLinePicture():
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