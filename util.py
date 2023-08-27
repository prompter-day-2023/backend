from dotenv import load_dotenv
import openai
import os

load_dotenv()

openai.api_key = os.getenv('GPT_API_KEY')

def get_images_from_dalle(gpt_result):
    dalle_prompt = convert_to_Dalle_prompt_from(gpt_result)

    # Dall-E 이미지 생성
    response = openai.Image.create(
        prompt = dalle_prompt,
        n = 1,    # 한 번에 생성할 이미지 개수 (test에는 1개로 진행합니다.)
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

# 한글 프롬프트를 영어 프롬프트로 번역, Dall-E 프롬프트에 맞게 가공하는 함수 (필요 시 사용)
def translate_gpt_prompt(message):
    url_for_deepl = 'https://api-free.deepl.com/v2/translate'
    payload = {
        'text': message,
        'source_lang': 'KO',
        'target_lang': 'EN'
    }
    headers = {
        'content-type': 'application/json',
        'Authorization': os.getenv('DEEPL_API_KEY')
    } 

    response = requests.post(url_for_deepl, json = payload, headers = headers)
    if response.status_code != 200:
        return { 'status': 557, 'message': '번역 생성에 실패하였습니다.' }
    data = response.json()

    idx = 0
    translate_result = ''
    line_length = len(data['translations'])

    for one_line in data['translations']:
        text = one_line['text']
        content_start_idx = text.find(':') + 2
        content = text[content_start_idx:]

        if idx == line_length - 1:
            translate_result += content[:-1]
        else:
            translate_result += content + ", "
        idx = idx + 1

    return translate_result