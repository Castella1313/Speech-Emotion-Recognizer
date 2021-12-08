import librosa
import soundfile
import os, glob, pickle
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time

import os
import pyaudio
import wave

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.clock import Clock
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

def extract_feature(file_name, mfcc, chroma, mel):    
    with soundfile.SoundFile(file_name) as sound_file:        
        X = sound_file.read(dtype="float32")        
        sample_rate=sound_file.samplerate        
        if chroma:            
            stft=np.abs(librosa.stft(X))        
            result=np.array([])        
        if mfcc:            
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)            
            result=np.hstack((result, mfccs))        
        if chroma:            
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)            
            result=np.hstack((result, chroma))        
        if mel:            
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)            
            result=np.hstack((result, mel))    
    return result

emotions={
     '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'}
observed_emotions=['calm', 'happy', 'fearful', 'disgust', 'sad', 'angry','surprised', 'neutral']

def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("./speech-emotion-recognition-ravdess-data/Actor_*/*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion in observed_emotions:
            feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        else: 
            continue
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

x_train,x_test,y_train,y_test=load_data(test_size=0.25)



model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

model.fit(x_train,y_train)
y_predicted = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predicted) * 100
print(accuracy)
def record_and_predict(self):
    print("* recording")
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()
    os.remove('./output.wav')
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'w')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    f = extract_feature("./output.wav", True, True, True).reshape(1,-1)
    pred_emotion = model.predict(f)[0]

    print(pred_emotion)
    return pred_emotion
    







class SER(App):
    def build(self):
        self.window = GridLayout()
        self.window.cols = 1
        self.window.size_hint = (0.6, 0.7)
        self.window.pos_hint = {"center_x": 0.5, "center_y":0.5}

        self.window.add_widget(Label(text="SPEECH EMOTION RECOGNIZER",
                                     font_size=30,))
        # image widget
        self.window.add_widget(Image(source="./e.jpg"))

        # label widget
        self.greeting = Label(
                        text= "Express yourself!!!",
                        font_size= 18,
                        color= '#00FFCE'
                        )
        self.window.add_widget(self.greeting)

        # button widget
        self.button = Button(
                      text= "Voice Input",
                      size_hint= (1,0.5),
                      bold= True,
                      background_color ='#00FFCE',
                      )
        self.button.bind(on_press=self.callback, on_release=self.releaseback)
        
        self.window.add_widget(self.button)

        return self.window
    def releaseback(self, instance):
        self.greeting.text = "Express Yourself!"
        self.button.text = "Voice Input"
        self.emot = self.find(self)
        self.greeting.text = "Your Emotion is: " + self.emot + "!"

    def find(self, instance):
        self.pred_emotion = record_and_predict(self)
        return self.pred_emotion
        
    def callback(self, instance):
        self.button.text = "Recording..."
        self.greeting.text = "Analysing audio"
        
        
if __name__ == "__main__":
    SER().run()