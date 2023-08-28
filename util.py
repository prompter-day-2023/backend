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

    image_url = { 'imageUrl': [] }
    idx = 0
    for list in response['data']:
        image_url['imageUrl'].append(list['url'])
        idx = idx + 1

    return image_url['imageUrl']

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
        return { 'status': 557, 'message': '번역 생성에 실패하였습니다.' }
    data = response.json()

    return data['translations']

    

def convert_trans_result_to_prompt(data_list):
    idx = 0
    translate_result = ''
    # line_length = len(data_list)

    for one_line in data_list:
        text = one_line['text']
        translate_result += text + "\n"
        # if idx == line_length - 1:
        #     translate_result += content[:-1]
        # else:
        #     translate_result += content + ", "
        idx = idx + 1
    return translate_result

def convert_trans_result_to_keyword_list(data_list):

    # data_str = data_str.split(",").strip()
    # print(data_str)
    keyword_list = []
    text = data_list[0]['text']
    keyword_list = text.split(", ")

    return keyword_list