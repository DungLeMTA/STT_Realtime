import time

import requests

url = "https://asr.hpda.vn/recog"

payload={'language': '2'}

headers = {}
def api_stt():
  files = [
    ('the_file', ('chunk2.wav', open(r'audio_temp.wav', 'rb'), 'audio/wav'))
  ]
  # start = time.time()
  response = requests.request("POST", url, headers=headers, data=payload, files=files, timeout=15)
  # end = time.time()
  # print("STT: %.3f"%(end-start))
  # print(response.text)
  return response.text

def api_stt_file(file):
  files = [
    ('the_file', ('chunk2.wav', open(file, 'rb'), 'audio/wav'))
  ]
  # start = time.time()
  response = requests.request("POST", url, headers=headers, data=payload, files=files, timeout=15)
  # end = time.time()
  # print("STT: %.3f"%(end-start))
  # print(response.text)
  return response.text


# print(api_stt())