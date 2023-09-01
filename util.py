from flask import abort
from dotenv import load_dotenv
import openai
import os
import requests

load_dotenv()

openai.api_key = os.getenv('GPT_API_KEY')
deepl_api_key = os.getenv('DEEPL_API_KEY')

def get_images_from_dalle(dalle_prompt):
    # Dall-E 이미지 생성
    response = openai.Image.create(
        prompt = dalle_prompt,
        n = 4,    # 한 번에 생성할 이미지 개수 (test에는 1개로 진행합니다.)
        size = '1024x1024'    # 256x256, 512x512, or 1024x1024 가능
    )

    if response is None:
        return 500, None

    image_url = { 'imageUrl': [] }
    idx = 0
    for list in response['data']:
        if list is None:
            return 500, None
        image_url['imageUrl'].append(list['url'])
        idx = idx + 1

    return 200, image_url['imageUrl']


def convert_to_Dalle_prompt_from(gpt_result):
    idx = 0
    result = ''
    category_list = ['Emotion', 'Characters', 'Picture color', 'Picture Context', 'One line summary']

    sentence_list = gpt_result.split('\n')
    line_length = len(sentence_list)
    
    if line_length == 0:
        return { 'code': 400, 'message': '프롬프트 결과물이 없습니다.' }
    
    for one_line in sentence_list:
        content_start_idx = one_line.find(':') + 2
        header = one_line[:content_start_idx-2]
        content = one_line[content_start_idx:]

        if idx == line_length - 1:
            result += content[:-1]
        else:
            result += content + ", "
        idx = idx + 1

    return 200, result


# 한글을 영어로 번역하는 함수
def translate_message(src_lang, tartget_lang, message):
    message_arr = []
    message_arr.append(message)
    url_for_deepl = 'https://api-free.deepl.com/v2/translate'
    payload = {
        'text': message_arr,
        'source_lang': src_lang,
        'target_lang': tartget_lang
    }
    headers = {
        'content-type': 'application/json',
        'Authorization': os.getenv('DEEPL_API_KEY')
    } 

    response = requests.post(url_for_deepl, json = payload, headers = headers)

    if response.status_code != 200:
        return 500, None
    data = response.json()

    return 200, data['translations']

def convert_trans_result_to_prompt(data_list):
    idx = 0
    translate_result = ''

    print('d', data_list)
    if len(data_list) == 0:
        return 500, None
    for one_line in data_list:
        text = one_line['text']
        translate_result += text + '\n'
        idx = idx + 1

    return 200, translate_result

def convert_trans_result_to_keyword_list(data_list):
    keyword_list = []
    text = data_list[0]['text']

    if text is None:
        return 500, None
    keyword_list = text.split(", ")

    return 200, keyword_list

