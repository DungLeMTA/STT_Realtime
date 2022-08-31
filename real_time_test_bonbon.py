
import pyaudio
import threading
import time
# from predict import *
import torchaudio
import torch
import wave
from API_STT_a_Dat import api_stt
import audioop
from collections import deque
import math
record_second = 3


class Listener:
    def __init__(self, sample_rate=16000, record_seconds=6, silence_limit = 1, silence_threshold=2000):
        self.chunk = 1024
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.p = pyaudio.PyAudio()
        self.silence_limit = silence_limit
        self.silence_threshold = silence_threshold
        self.index = 1
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  rate=self.sample_rate,
                                  channels=1,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=self.chunk)

    def listen(self, queue, prev_audio):

        # while True:
        #     data = self.stream.read(self.chunk, exception_on_overflow=True)
        #
        #     queue.append(data)
        #
        #     time.sleep(0.01)
        while True:

            listen = True
            started = False
            rel = self.sample_rate / self.chunk

            # prev_audio = deque(maxlen=int(2 * rel))
            slid_window = deque(maxlen=int(self.silence_limit * rel))

            while listen:
                data = self.stream.read(self.chunk, exception_on_overflow=True)
                slid_window.append(math.sqrt(abs(audioop.avg(data, 4))))

                if (sum([x > self.silence_threshold for x in slid_window]) > 0):
                    if (not started):
                        # print("Starting record of phrase " + str(self.index))
                        # self.index += 1
                        started = True
                elif (started is True):
                    started = False
                    listen = False
                    # print("End record ")
                    prev_audio = deque(maxlen=int(1 * rel))

                if (started is True):
                    # queue = list(prev_audio)+queue
                    queue.append(data)

                    # print(done)
                else:
                    prev_audio.append(data)

            time.sleep(0.1)

    def run(self, queue, pre, start_):
        thread = threading.Thread(target=self.listen, args=(queue, pre, start_,), daemon=False)
        thread.start()
        print('\ Speech recognition engine is now listening...')

class SpeechRecognitionEngine:
    def __init__(self):
        self.listener = Listener(sample_rate=16000)

        self.audio_q = list()
        self.prev = deque(maxlen=int(2 * 16000/1024))

        self.out_arg = None
        self.lenght_wav = None
        self.start = None
        # self.context_length = context_length * 50  # multiply by 50 because each 50 from output frame is 1 second

    def save(self, waveforms, prev,fname="audio_temp.wav"):
        wf = wave.open(fname, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(self.listener.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(list(prev)))
        wf.writeframes(b"".join(waveforms))
        wf.close()
        return fname

    def save_2(self, waveforms, prev,index):
        fname = "sound_save/audio_temp_" + str(index) + ".wav"
        index += 1
        wf = wave.open(fname, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(self.listener.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(list(prev)))
        wf.writeframes(b"".join(waveforms))
        wf.close()
        return fname

    def predict(self, audio,prev,index):
        with torch.no_grad():
            fname_2 = self.save_2(audio,prev,index)
            fname = self.save(audio,prev)
            waveform, _ = torchaudio.load(fname)  # don't normalize on train
            # print(waveform)
            # waveform = features(waveform,sampling_rate=16000,return_tensors="pt").input_values.to(device)
            # print(waveform)
            self.lenght_wav = waveform if self.lenght_wav is None else torch.cat((self.lenght_wav, waveform), dim=1)
            # print('leng_wav: ',self.lenght_wav)
            results = api_stt()

            current_context_length = self.lenght_wav.shape[1] / 160000  # in seconds
            # print(current_context_length)
            if current_context_length > 5:
                self.lenght_wav = None
            return results, current_context_length

    def send_data(self, action, index):
        print('sending..')
        prev = self.prev.copy()
        pred_q = self.audio_q.copy()
        self.audio_q.clear()
        action(self.predict(pred_q, prev, index))
        index += 1
        del pred_q
        del prev

    def inference_loop(self, action):
        index = 1
        while True:

            if len(self.audio_q) < 5:
                continue
            else:
                pre = self.prev.copy()
                pred_q = self.audio_q.copy()
                self.audio_q.clear()
                action(self.predict(pred_q, pre, index))
                index += 1
                del pred_q
            time.sleep(record_second)

    def run(self, action):
        self.listener.run(self.audio_q,self.prev,self.start)
        thread = threading.Thread(target=self.inference_loop, args=(action,), daemon=False)
        thread.start()

class DemoAction:

    def __init__(self):
        self.asr_results = ""
        self.current_beam = ""

    def __call__(self, x):
        # print('x:',x)
        results, current_context_length = x
        self.current_beam = results
        # print('asr: ',self.asr_results)
        # print('result: ',results)
        trascript = self.asr_results+' '+results
        # if results.strip() != '':
        #     print(results)
        # if current_context_length > 1 and results.strip() != '':
        if results.strip() != '':
            self.asr_results = trascript
            print('ASR_RESULT: ',self.asr_results)
            print('===========')
            print('------------------')

import os
def Init(folder1):
    try:
        for file in os.listdir(folder1):
            os.remove(folder1+file)
    except:
        pass

if __name__ == "__main__":
    Init('./sound_save/')
    index = 1
    asr_engine = SpeechRecognitionEngine()
    action = DemoAction()

    asr_engine.run(action)
    threading.Event().wait()
