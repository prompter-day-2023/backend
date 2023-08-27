from flask import Flask
from flask import jsonify
import cv2
import numpy as np
import openai
import os
from datetime import datetime
from dotenv import load_dotenv
import requests
load_dotenv()

app = Flask(__name__)

openai.api_key = os.getenv('GPT_API_KEY')

@app.route('/diary', methods=['POST'])
def create_diary():
    contents = request.json.get('contents')
    commend = f"Based on the diary contents written by the child, please write the diary contents and situation in English according to the format below. The purpose is to create an image by putting a prompt into the generative AI.\n\nEmotion:\nSubject:\nPicture color:\nOne line summary:\n\nThe diary contains the following.\n{contents}"

    response = openai.Completion.create(
        model = 'text-davinci-003',   # openai에서 제공하는 모델 입력 (GPT-3.5)
        prompt = commend,  # 원하는 실행어 입력
        temperature = 0,
        max_tokens = 300,   # 입력 + 출력 값으로 잡을 수 있는 max_tokens 값
        frequency_penalty = 0.0,
        presence_penalty = 0.0
    )

    dalle_prompt = response.choices[0].text.strip()

    # TODO: DALLE 연결하기

    return { "response": response.choices[0].text.strip() }

def create_line_picture(image):
    image_name = image.filename.split('.')[0]
    image_type = image.filename.split('.')[-1]
    created_at = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    new_file_name = f"{image_name}-{created_at}.{image_type}"

    # 파일 데이터를 읽어와 NumPy 배열로 변환
    image_data = image.read()
    np_array = np.frombuffer(image_data, np.uint8)
    np_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    gray_img = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
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