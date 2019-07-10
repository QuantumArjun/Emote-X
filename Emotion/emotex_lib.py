#!/usr/bin/env python
# coding: utf-8

# # Dependencies

# In[1]:

__name__ = 'emotex_lib'
import scipy
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
import osascript
from gtts import gTTS 
import os 
import pyaudio
import wave
import keyboard  # using module keyboard
import soundfile as sf
import math
import pyloudnorm as pyln
from sys import byteorder
from array import array
from struct import pack
import librosa
from scipy.signal import butter, sosfiltfilt
import python_speech_features
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
import pysptk
from  conch.analysis.formants import lpc


# # Constants

# In[ ]:


BANDPASS_FREQ = [300, 3400]


# # Get and Process Sound Dataset

# In[ ]:


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', analog=False, output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y


# In[ ]:


def remove_silence_from(amplitudes, threshold):
    silenced = []
    for x in amplitudes:
        if x >= threshold:
            silenced.append(x)
    return silenced


# In[ ]:


def extract_data(file_location):
    BANDPASS_FREQ = [300, 3400]
    fs, data = wavfile.read(file_location)
    number_of_samples = data.shape[0]
    meta_data = open(r"LDC2002S28band-txt.txt")
    meta_data = pd.read_csv("LDC2002S28-txt.txt", sep="A:", header=None, engine='python')
    meta_data.columns = ["sound limits","description"]
    
    #dual channel to one channel
    data = np.average(data, axis = 1)
    #remove noise
    data = butter_bandpass_filter(data, BANDPASS_FREQ[0], BANDPASS_FREQ[1], fs)
    
    # removee extra data pointts
    meta_data = meta_data[meta_data.description != ' [MISC]']
    meta_data = meta_data[~meta_data['description'].astype(str).str.startswith(' (')]
    meta_data = meta_data[~meta_data['description'].astype(str).str.startswith(' Emotion category elation')]
    meta_data = meta_data[~meta_data['description'].astype(str).str.startswith('  [MISC]')]

    # description and time limits 
    voice_time_limits = meta_data["sound limits"]
    voice_time_limits = [i.split(" ")[0:2] for i in voice_time_limits]
    voice_time_limits = np.array(voice_time_limits)
    voice_time_limits = voice_time_limits.astype(np.float)
    description = meta_data["description"]
    description = [i.split(",")[0].strip() for i in description]

    #divide the dataa set
    divided_data = []
    for i in voice_time_limits:
        startingpoint = int(i[0]*fs)
        endingpoint = int(i[1]*fs)
        divided_data.append(data[startingpoint:endingpoint])
    np_data = np.asarray(divided_data)
    return np_data, description, len(divided_data), fs


# #### Split Dataset

# # Feature Extraction

# In[ ]:


def MFCC_algorithm(np_data, fs):
        # MFCC function taking the first thirteen coef
    MFCC2 = []
    for i in np_data:
        i = np.asarray(i)
        MFCC2.append(python_speech_features.base.mfcc(i, samplerate=fs, 
                                     winlen=0.025, winstep=0.01, numcep=13, 
                                     nfilt=26, nfft=552))
        
    # gather information from the MFCC (feature extraction)
    MFCC3 = []
    cache = {}
    for data_point in MFCC2:
        for time_segment in data_point:
            if (data_point[0] == time_segment).all():
                for i in range(13):
                    cache[i] = [time_segment[i]]
            else:
                for i in range(13):
                    cache[i] = np.concatenate((cache[i], [time_segment[i]]))
        cached_variables = []
        cache_grad = []
        for i in range(13):
            cache_grad.append(np.gradient(cache[i]))
            cached_variables.append([np.mean(cache[i]), np.median(cache[i]), np.var(cache[i]), 
                               np.min(cache[i]), np.max(cache[i]), 
                                     np.mean(cache_grad[i]), np.var(cache_grad[i])])
        MFCC3.append(np.hstack(np.hstack(cached_variables)))
    return MFCC3
    


# In[ ]:


def get_pitch_vector(data, fs):
    data = np.float32(data)
    pitch = pysptk.sptk.rapt(data, fs, hopsize = 50)
    silenced = remove_silence_from(pitch, np.mean(pitch))
    return silenced

def get_pitch_stats(np_array, fs):
    stats_matrix = []
    for data in np_array:
        pitch_vector = get_pitch_vector(data, fs)
        stats = get_stats(pitch_vector)
        stats_matrix.append(stats)
    return stats_matrix


# In[ ]:


def get_spectral_vector(data, fs):
    data = np.float32(data)
    cent = librosa.feature.spectral_centroid(y=data, sr=fs)
    return cent
def get_spectral_stats(np_array, fs):
    stats_matrix = []
    for data in np_array:
        spectral_vector = get_spectral_vector(data, fs)
        stats = get_stats(spectral_vector)
        stats_matrix.append(stats)
    return stats_matrix


# In[ ]:


def get_lpc_vector(data):
    vec = lpc.lpc_ref(data, 12)
    return vec
def get_lpc_stats(np_array):
    stats_matrix = []
    for data in np_array:
        lpc_vector = get_lpc_vector(data)
        stats_matrix.append(lpc_vector[1:])  #remove the first number, it's not useful
    return stats_matrix


# In[ ]:


def get_rms_vector(data):
    temp_data = np.float32(data)
    cent = librosa.feature.rms(y=temp_data)
    return cent
def get_rms_stats(np_array):
    stats_matrix = []
    for data in np_array:
        rms_vector = get_rms_vector(data)
        stats = get_stats(rms_vector)
        stats_matrix.append(stats)
    return stats_matrix


# In[ ]:


def get_zero_vector(data):
    temp_data = np.float32(data)
    cent = librosa.feature.zero_crossing_rate(y=temp_data)
    return cent
def get_zero_stats(np_array):
    stats_matrix = []
    for data in np_array:
        zero_vector = get_zero_vector(data)
        stats = get_stats(zero_vector)
        stats_matrix.append(stats)
    return stats_matrix


# In[ ]:


def get_sr_vector(data):
    temp_data = np.float32(data)
    cent = librosa.feature.spectral_rolloff(y=temp_data)
    return cent
def get_sr_stats(np_array):
    stats_matrix = []
    for data in np_array:
        sr_vector = get_sr_vector(data)
        stats = get_stats(sr_vector)
        stats_matrix.append(stats)
    return stats_matrix


# In[ ]:


def get_stats(pitch_vector):
    mean = np.mean(pitch_vector)
    median = np.median(pitch_vector)
    low = np.min(pitch_vector)
    high = np.max(pitch_vector)
    variance = np.var(pitch_vector)
    
    #derivative
    derivative = np.diff(pitch_vector)
    d_mean = np.mean(derivative)
    d_min = np.min(derivative)
    d_max = np.max(derivative)
    return [mean, median, low, high, variance, d_mean, d_min, d_max]


# # emotional extraction
# given an array of the emotions it converts the array to a number, if an emotion is not there it will print it out and break the loop 

# In[ ]:



def emotion_extraction(description, number_examples):
    nu_emotion = 15
    y = np.zeros(shape=(nu_emotion, number_examples))
    counter = 0
    for i in description:
        X0 = np.zeros((number_examples,1))
        if i == 'neutral':
            y[0][counter] = 1
        elif i == 'disgust':
             y[1][counter] = 1
        elif i == 'panic':
             y[2][counter] = 1
        elif i == 'anxiety':
             y[3][counter] = 1
        elif i == 'hot anger':
             y[4][counter] = 1
        elif i == 'cold anger':
             y[5][counter] = 1
        elif i == 'despair':
             y[6][counter] = 1
        elif i == 'sadness':
             y[7][counter] = 1
        elif i == 'elation':
             y[8][counter] = 1
        elif i == 'happy':
             y[9][counter] = 1
        elif i == 'interest':
             y[10][counter] = 1
        elif i == 'boredom':
             y[11][counter] = 1
        elif i == 'shame':
             y[12][counter] = 1
        elif i == 'pride':
             y[13][counter] = 1
        elif i == 'contempt':
             y[14][counter] = 1
        else:
            print(i)
            break
        counter +=1
    y = np.transpose(y)
    return y


# # Ready Dataset and output
# Put all of the extracted features into X and the classifications into y and split into training and testing group

# In[67]:


def x_y_split(filepath):
    data, description, data_len, fs = extract_data(filepath)
    x = MFCC_algorithm(data, fs)
    x1 = get_pitch_stats(data, fs)
    x2 = get_spectral_stats(data, fs)
    x3 = get_lpc_stats(data)
    x4 = get_rms_stats(data)
    x5 = get_sr_stats(data)
    x6 = get_zero_stats(data)
    x = np.concatenate((x,x1,x2,x3,x4, x5, x6), axis=1)
    y = emotion_extraction(description, data_len)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    num_labels = y_train.shape[1]
    num_features = X_train.shape[1]
    print("x train shape: " +str(X_train.shape))
    print("y train shape: " +str(y_train.shape))
    print("x test shape: " +str(X_test.shape))
    print("y test shape: " +str(y_test.shape))
    for i in range(num_labels):
        print("y_train for emotion "+str(i)+": "+ str(np.sum(y_train[:,i])))
    for i in range(num_labels): 
        print("y_test for emotion "+str(i)+": "+ str(np.sum(y_test[:,i])))
    return X_train, X_test, y_train, y_test


# In[68]:


X_train, X_test, y_train, y_test = x_y_split('../../LDC2002S28.wav')


# In[ ]:




