from flask import Flask
from flask import jsonify, request
from datetime import datetime
from dotenv import load_dotenv
from .s3_bucket import s3
import cv2
import numpy as np
import openai
import os
import wget

load_dotenv()

app = Flask(__name__)

openai.api_key = os.getenv('GPT_API_KEY')
bucket_name = os.getenv('BUCKET_NAME')
bucket_url_prefix = os.getenv('BUCKET_URL_PREFIX')

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
    # dalle 사진의 경우 result 롤더, 라인드로잉 사진의 경우 line 폴더 사용
    s3.put_object(
            Body = file,
            Bucket = bucket_name,
            Key = f'result/{file_name}.{file_type}',
            ContentType = f'image/{file_type}'
    )
    return f'{bucket_url_prefix}/result/{file_name}.{file_type}'

@app.route('/convert-test', methods=['POST'])
def create_line_picture():
    image_url = request.json.get('imageUrl')
    # TODO: 이미지 파일이 backend에 쌓이는 문제 발생 -> 성능 개선 필요
    download_image = wget.download(image_url)
    image_file = cv2.imread(download_image)

    # 파일명 생성하기
    split_url = image_url.split('/')[-1]
    image_name = split_url.split('.')[0]
    image_type = split_url.split('.')[-1]
    created_at = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    new_file_name = f"{image_name}-{created_at}.{image_type}"

    gray_img = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray_img, 100, 250)

    # 엣지 확장을 위한 커널 생성
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=1)

    # 배경색과 엣지색을 지정
    background_color = (255, 255, 255)  # 흰색 배경
    edge_color = (0, 0, 0)  # 검은색 엣지

    # 배경 부분을 원하는 배경색으로 채우기
    filled_image = np.full_like(image_file, background_color)
    filled_image[dilated != 0] = edge_color

    cv2.imwrite(new_file_name, filled_image)

    data = cv2.imencode(f'.{image_type}', filled_image)[1].tobytes()

    s3.put_object(
            Body = data,
            Bucket = bucket_name,
            Key = f'line/{image_name}.{image_type}',
            ContentType = f'image/{image_type}'
    )
    return f'{bucket_url_prefix}/line/{image_name}.{image_type}'


if __name__ == '__main__':
    app.run(debug=True, port=5000)