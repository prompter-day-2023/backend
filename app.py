from flask import Flask
from flask import jsonify, request
import cv2
import numpy as np
import openai
import os
from datetime import datetime
from dotenv import load_dotenv
from .s3_bucket import s3

load_dotenv()

app = Flask(__name__)

openai.api_key = os.getenv('GPT_API_KEY')
bucket_name = os.getenv('BUCKET_NAME')

@app.route('/diary', methods=['POST'])
def create_diary():
    contents = request.json.get('contents')
    command = f'Based on the diary contents written by the child, please write the diary contents and situation in English according to the format below. The purpose is to create an image by putting a prompt into the generative AI.\n\nEmotion:\nSubject:\nPicture color:\nOne line summary:\n\nThe diary contains the following.\n{contents}'

    response = openai.Completion.create(
        model = 'text-davinci-003',   # openai에서 제공하는 모델 입력 (GPT-3.5)
        prompt = command,  # 원하는 실행어 입력
        temperature = 0,
        max_tokens = 300,   # 입력 + 출력 값으로 잡을 수 있는 max_tokens 값
        frequency_penalty = 0.0,
        presence_penalty = 0.0
    )

    dalle_prompt = response.choices[0].text.strip()

    # TODO: DALLE 연결하기

    return { "response": response.choices[0].text.strip() }

# S3 테스트 코드입니다. 향후 코드 작성 시 참고해주세요.
@app.route('/s3-test')
def upload_s3():
    file = request.files['file']
    file_name = file.filename.split('.')[0]
    file_type = file.filename.split('.')[-1]

    # 사진을 s3에 저장
    s3.put_object(
            Body = file,
            Bucket = bucket_name,
            Key = f'result/{file_name}.{file_type}',
            ContentType = f'image/{file_type}'
    )
    return 'success'

def create_line_picture(image):
    # TODO: s3 url을 다운받도록 수정
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

    byte_image_to_s3 = cv2.imencode(image_type, filled_image)[1].tobytes()


if __name__ == '__main__':
    app.run(debug=True, port=5000)