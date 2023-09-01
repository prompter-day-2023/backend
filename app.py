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
from botocore.exceptions import ClientError

load_dotenv()

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv('GPT_API_KEY')
bucket_name = os.getenv('BUCKET_NAME')
expire_time = os.getenv('PICTURE_EXPIRE_TIME')

@app.route('/diary', methods=['POST'])
def create_diary():
    title = request.json.get('title')
    contents = request.json.get('contents')

    # 일기 내용 -> 영어로 번역
    diary_trans_input = f'제목: {title}\n{contents}'
    diary_trans_result = util.translate_message('KO', 'EN', diary_trans_input)
    contents_eng = util.convert_trans_result_to_prompt(diary_trans_result)

    command = f"Based on the diary contents written by an adult, please write the diary contents and situation in English according to the format below. The purpose is to create an image by putting a prompt into the generative AI. Be sure to include a 'one-line summary' and 'Picture Context' has no more than 10 words of  context.\nEmotion:\nCharacters:\nPicture color:\nPicture Context:\nOne line summary in 10 words:The diary contains the following.{contents_eng}"

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
    dalle_prompt += ', simple vector illustration'
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

        get_url = s3_bucket.s3.generate_presigned_url('get_object', Params = { 'Bucket': bucket_name, 'Key': f'result/{image_name}.{image_type}' }, ExpiresIn = expire_time)
        image_url_list.append(get_url)
        
    data = { 'image_url': image_url_list, 'keywords': keyword_list }

    return { 'data': data, 'code': 200, 'message': '이미지 생성에 성공하였습니다.' }


@app.route('/line-drawing', methods=['POST'])
def create_line_picture():
    image_url = request.json.get('imageUrl')
    download_image = requests.get(image_url)
    image_data = np.frombuffer(download_image.content, np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    # 파일명 생성하기
    split_url = image_url.split('?')[0].split('/')[-1]
    image_name = split_url.split('.')[0]
    image_type = split_url.split('.')[-1]

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobel = cv2.Sobel(gray_img, cv2.CV_8U, 1, 0, 3)
    laplacian = cv2.Laplacian(gray_img, cv2.CV_8U, ksize=3)
    canny = cv2.Canny(image, 30, 40)

    # 엣지 확장을 위한 커널 생성
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=1)

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

    get_url = s3_bucket.s3.generate_presigned_url('get_object', Params = { 'Bucket': bucket_name, 'Key': f'line/{image_name}.{image_type}' }, ExpiresIn = expire_time)

    return { 'data': get_url, 'code': 200, 'message': '이미지 생성에 성공하였습니다.' }
    
if __name__ == '__main__':
    app.run(debug=True, port=5123)