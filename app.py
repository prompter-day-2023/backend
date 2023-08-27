from flask import Flask, jsonify, request
import cv2
import numpy as np
import openai
import os
from datetime import datetime
from dotenv import load_dotenv
import requests
from .s3_bucket import s3
import wget

load_dotenv()

app = Flask(__name__)

openai.api_key = os.getenv('GPT_API_KEY')
bucket_name = os.getenv('BUCKET_NAME')
bucket_url_prefix = os.getenv('BUCKET_URL_PREFIX')

@app.route('/diary', methods=['POST'])
def create_diary():
    contents = request.json.get('contents')
    command = f'Based on the diary contents written by the child, please write the diary contents and situation in English according to the format below. The purpose is to create an image by putting a prompt into the generative AI.\n\nEmotion:\nCharacters:\nPicture color:\nOne line summary:\n\nThe diary contains the following.\n{contents}'

    response = openai.Completion.create(
        model = 'text-davinci-003',   # openai에서 제공하는 모델 입력 (GPT-3.5)
        prompt = command,  # 원하는 실행어 입력
        temperature = 0,
        max_tokens = 300,   # 입력 + 출력 값으로 잡을 수 있는 max_tokens 값
        frequency_penalty = 0.0,
        presence_penalty = 0.0
    )

    gpt_result = response.choices[0].text.strip()
    dalle_url_list = get_images_from_dalle(gpt_result)

    image_url_list = []
    for url in dalle_url_list:    
        download_image = wget.download(url)
        image_file = cv2.imread(download_image)

        split_url = url.split('/')[6]
        image_full_name = split_url.split('?')[0]
        image_name = image_full_name.split('.')[0]
        image_type = image_full_name.split('.')[-1]

        data = cv2.imencode(f'.{image_type}', image_file)[1].tobytes()

        s3.put_object(
            Body = data,
            Bucket = bucket_name,
            Key = f'result/{image_name}.{image_type}',
            ContentType = f'image/{image_type}'
        )   

        image_url_list.append(f'{bucket_url_prefix}/result/{image_name}.{image_type}')

    
    return { "response": image_url_list }


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


def get_images_from_dalle(gpt_result):
    dalle_prompt = convert_to_Dalle_prompt_from(gpt_result)

    # Dall-E 이미지 생성
    openai.api_key = os.getenv('GPT_API_KEY')
    response = openai.Image.create(
        prompt=dalle_prompt,
        n=1,    # 한 번에 생성할 이미지 개수 (test에는 1개로 진행합니다.)
        size="1024x1024"    # 256x256, 512x512, or 1024x1024 가능
    )

    image_url = {"imageUrl": []}
    idx = 0
    for list in response['data']:
        image_url["imageUrl"].append(list['url'])
        idx = idx + 1

    return image_url["imageUrl"]


def convert_to_Dalle_prompt_from(gpt_result):
    sentence_list = gpt_result.split('\n')
    idx = 0
    result = ''
    line_length = len(sentence_list)

    for one_line in sentence_list:
        content_start_idx = one_line.find(":") + 2
        content = one_line[content_start_idx:]
        if idx == line_length - 1:
            result += content[:-1]
        else:
            result += content + ", "
        idx = idx + 1
    result += ', vector illustration'

    return result
    

# 한글 프롬프트를 영어 프롬프트로 번역, Dall-E 프롬프트에 맞게 가공하는 함수 (필요 시 사용)
def translate_gpt_prompt(message):
    url_for_deepl = 'https://api-free.deepl.com/v2/translate'
    payload = {
        'text': message,
        'source_lang': 'KO',
        'target_lang': 'EN'
    }

    headers = {
        "content-type": "application/json",
        "Authorization": os.getenv('DEEPL_API_KEY')
    } 

    response = requests.post(url_for_deepl, json=payload, headers=headers)
    if response.status_code != 200:
        return { "status": 557, "message": '번역 생성에 실패하였습니다.' }
    data = response.json()

    idx = 0
    translate_result = ''
    line_length = len(data['translations'])

    for one_line in data['translations']:
        text = one_line['text']
        content_start_idx = text.find(":") + 2
        content = text[content_start_idx:]

        if idx == line_length - 1:
            translate_result += content[:-1]
        else:
            translate_result += content + ", "
        idx = idx + 1

    return translate_result
    


if __name__ == '__main__':
    app.run(debug=True, port=5000)