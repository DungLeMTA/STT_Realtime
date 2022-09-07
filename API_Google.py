import speech_recognition as sr
import os
for file in os.listdir('./sound_save'):
    file_audio = './sound_save/'+file
    text = sr.Recognizer.recognize_google(file_audio, language="en")
    print(text)