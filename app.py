from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, jsonify, request
import s3_bucket
import util
from flask_cors import CORS
import cv2
import numpy as np
import openai
import os
import requests
import wget

load_dotenv()

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv('GPT_API_KEY')
bucket_name = os.getenv('BUCKET_NAME')
bucket_url_prefix = os.getenv('BUCKET_URL_PREFIX')

@app.route('/diary', methods=['POST'])
def create_diary():
    title = request.json.get('title')
    contents = request.json.get('contents')

    # 일기 내용 -> 영어로 번역
    diary_trans_input = "제목: " + title + "\n" + contents
    diary_trans_result = util.translate_message('KO', 'EN', diary_trans_input)
    contents_eng = util.convert_trans_result_to_prompt(diary_trans_result)

    command = f'Based on the diary contents written by the child, please write the diary contents and situation in English according to the format below. The purpose is to create an image by putting a prompt into the generative AI.\n\nEmotion:\nCharacters:\nPicture color:\nOne line summary:\n\nThe diary contains the following.\n{contents_eng}'

    response = openai.Completion.create(
        model = 'text-davinci-003',   # openai에서 제공하는 모델 입력 (GPT-3.5)
        prompt = command,  # 원하는 실행어 입력
        temperature = 0,
        max_tokens = 1500,   # 입력 + 출력 값으로 잡을 수 있는 max_tokens 값
        frequency_penalty = 0.0,
        presence_penalty = 0.0
    )

    gpt_result = response.choices[0].text.strip()

    # gpt 영어 결과 -> Dalle 프롬프트 가공
    dalle_prompt = util.convert_to_Dalle_prompt_from(gpt_result)

    # # # Dalle 프롬프트 -> 조회할 요약된 키워드로 번역
    dalle_prompt_trans_result = util.translate_message('EN', 'KO', dalle_prompt)
    keyword_list = util.convert_trans_result_to_keyword_list(dalle_prompt_trans_result)

    # 한줄 요약 문장은 키워드 리스트에서 제외
    keyword_list.pop()

    # # Dalle 이미지 생성
    dalle_prompt += ', vector illustration'
    dalle_url_list = util.get_images_from_dalle(dalle_prompt)

    image_url_list = []
    for url in dalle_url_list:    
        download_image = requests.get(url)
        image_data = np.frombuffer(download_image.content, np.uint8)
        image_file = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        split_url = url.split('/')[6]
        image_full_name = split_url.split('?')[0]
        image_name = image_full_name.split('.')[0]
        image_type = image_full_name.split('.')[-1]

        data = cv2.imencode(f'.{image_type}', image_file)[1].tobytes()

        s3_bucket.s3.put_object(
            Body = data,
            Bucket = bucket_name,
            Key = f'result/{image_name}.{image_type}',
            ContentType = f'image/{image_type}'
        )   

        image_url_list.append(f'{bucket_url_prefix}/result/{image_name}.{image_type}')

    data = {"image_url": image_url_list, "keywords": keyword_list}
    return {"data": data, "code": 200, "message": "이미지 생성에 성공하였습니다." }


@app.route('/line-drawing', methods=['POST'])
def create_line_picture():
    image_url = request.json.get('imageUrl')
    download_image = requests.get(image_url)
    image_data = np.frombuffer(download_image.content, np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    # 파일명 생성하기
    split_url = image_url.split('/')[-1]
    image_name = split_url.split('.')[0]
    image_type = split_url.split('.')[-1]
    created_at = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    new_file_name = f"{image_name}-{created_at}.{image_type}"

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

    data = cv2.imencode(f'.{image_type}', filled_image)[1].tobytes()

    s3_bucket.s3.put_object(
            Body = data,
            Bucket = bucket_name,
            Key = f'line/{image_name}.{image_type}',
            ContentType = f'image/{image_type}'
    )
    return { 'response': f'{bucket_url_prefix}/line/{image_name}.{image_type}' }
    
if __name__ == '__main__':
    app.run(debug=True, port=5123)