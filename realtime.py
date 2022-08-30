
import wave
import pyaudio
import threading
import time
import argparse
import torchaudio 
import sys
import numpy as np
from predict import *
import wave
from API_STT_a_Dat import api_stt

class Listener:
    def __init__(self,sample_rate = 16000,record_seconds=2):
        self.chunk = 1024
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  rate=self.sample_rate,
                                  channels=1,
                                  input = True,
                                  output=True,
                                  frames_per_buffer=self.chunk)
    def listen(self,queue):
        while True:
            data = self.stream.read(self.chunk,exception_on_overflow= False)
            queue.append(data)
            time.sleep(0.01)
    def run(self,queue):
        thread = threading.Thread(target=self.listen,args=(queue,),daemon=True)
        thread.start()
        print('\ Speech recognition engine is now listening...')

class SpeechRecognitionEngine:
    def __init__(self,context_length=10):
        self.listener = Listener(sample_rate=16000)
        self.model = model
        self.featurizer = processor.feature_extractor
        self.audio_q = list()
        self.out_arg = None
        self.lenght_wav =None
        self.start = False
        self.context_length = context_length * 50 # multiply by 50 because each 50 from output frame is 1 second
    def save(self,pre,waveforms,fname="audio_temp"):
        wf = wave.open(fname,"wb")
        wf.setnchannels(1)
        wf.setsampwidth(self.listener.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b"".join(waveforms))
        wf.close()
        return fname
    def predict(self,audio):
        with torch.no_grad():
            fname = self.save(audio)
            waveform, _ = torchaudio.load(fname) # don't normalize on train

            # waveform = features(waveform,sampling_rate=16000,return_tensors="pt").input_values.to(device)
            # print(waveform)
            self.lenght_wav = waveform if self.lenght_wav is None else torch.cat((self.lenght_wav,waveform),dim=1)
            input = processor.feature_extractor(self.lenght_wav[0],sampling_rate=16000,return_tensors="pt")#.input_values.to(device)
            # input = processor.feature_extractor(waveform[0], sampling_rate=16000, return_tensors='pt')
            if torch.cuda.is_available():
                input = {key: value.cuda() for key, value in input.items()}
            output = model(**input)
            logits = output.logits
            self.out_arg = logits[0] #if self.out_arg is None else torch.cat((self.out_arg,logits[0]),dim=0)
            # ngram_lm_model = get_decoder_ngram_model(tokenize,lm_file)
            # results = ngram_lm_model.decode(self.out_arg.detach().numpy(), beam_width=500)
            results=processor.decode(output.logits.cpu().detach().numpy()[0], beam_width=100)
            # predicted_ids = torch.argmax(logits, dim=-1)
            # results = tokenize.batch_decode(predicted_ids)
            current_context_length = self.lenght_wav.shape[1] / 160000 #in seconds
            # print(current_context_length)
            if current_context_length > 4:
                self.lenght_wav = None
            return results[0],current_context_length
    def inference_loop(self,action):
        while True:     
            if len(self.audio_q) < 5 :
                continue
            else:
                pred_q = self.audio_q.copy()
                self.audio_q.clear()
                action(self.predict(pred_q))
                del pred_q
            time.sleep(0.01)
            
    def run(self,action):
        self.listener.run(self.audio_q)
        thread = threading.Thread(target=self.inference_loop, args=(action,),daemon=True)
        thread.start()
    
    
        
class DemoAction:

    def __init__(self):
        self.asr_results = ""
        self.current_beam = ""

    def __call__(self, x):
        results, current_context_length = x
        self.current_beam = results
        trascript = " ".join(self.asr_results.split() + results.split())
        print(trascript)
        if current_context_length > 10:
            self.asr_results = trascript

if __name__ == "__main__":
     asr_engine = SpeechRecognitionEngine()
     action = DemoAction()
     
     asr_engine.run(action)
     threading.Event().wait()     
