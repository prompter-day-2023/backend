from flask import Flask
from flask import request, jsonify
import cv2
import numpy as np
from datetime import datetime
import requests
import openai
import os

app = Flask(__name__)

@app.route('/diary', methods=['POST'])
def getImages():
    # 문자열->문자열 배열에 넣어주는 처리
    message = request.form['text']
    message = message.split('\n')

    # dalle_prompt = translateGptPrompt(message)
    dalle_prompt += ', vector illustration'

    # Dall-E 이미지 생성
    openai.api_key = os.getenv('GPT_API_KEY')
    response = openai.Image.create(
        prompt=dalle_prompt,
        n=3,
        size="1024x1024"    # 256x256, 512x512, or 1024x1024 가능
    )

    image_url = {"imageUrl": []}
    idx = 0
    for list in response['data']:
        image_url["imageUrl"].append(list['url'])
        idx = idx + 1

    print(image_url["imageUrl"])

    return { "status": 200, "message": 'OK', "data": image_url }


# 한글 프롬프트를 영어 프롬프트로 번역, Dall-E 프롬프트에 맞게 가공하는 함수
def translateGptPrompt(message):
    url_for_deepl = 'https://api-free.deepl.com/v2/translate'
    payload = {
        'text': message,
        'source_lang': 'KO',
        'target_lang': 'EN'
    }

    #  todo: .env에 key 보관
    headers = {
        "content-type": "application/json",
        "Authorization": os.getenv('DEEPL_API_KEY')
    } 

    response = requests.post(url_for_deepl, json=payload, headers=headers)
    if response.status_code != 200:
        return { "status": 557, "message": '번역 생성에 실패하였습니다.' }
    data = response.json()

    print(data)

    idx = 0
    translate_result = ''
    line_length = len(data['translations'])

    for one_line in data['translations']:
        text = one_line['text']
        content_start_idx = text.find(":") + 2
        content = text[content_start_idx:]
        # print(content)
        if idx == line_length - 1:
            translate_result += content[:-1]
        else:
            translate_result += content + ", "
        idx = idx + 1

    return translate_result
    


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