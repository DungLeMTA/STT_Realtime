from API_STT_a_Dat import api_stt_file
import os
i=1
for file in os.listdir('./sound_save'):
    file_audio = './sound_save/audio_temp_'+str(i)+'.wav'
    text = api_stt_file(file_audio)
    print(i,': ',text)
    i+=1