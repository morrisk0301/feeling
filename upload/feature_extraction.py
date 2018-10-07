# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 18:09:06 2018

@author: Kyeong In Kim
"""

import librosa
import essentia
import essentia.streaming
import numpy as np
import pandas as pd
from essentia.standard import *
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from specSlope import *

#file open
file_name = "ang_2"
file_path = 'sound_anger/'+file_name+'.wav'
spectral_path = "result_python/spectral/"+file_name+".csv"
harmonic_path = "result_python/harmonic/"+file_name+".csv"


#Variable Declaration
specSlope_essentia = []
specSpread_essentia = []
harEnergy_essentia = []
noiseEnergy_essentia = []
noisiness_essentia = []
inharmonicity_essentia = []
tristimulus1_essentia = []
tristimulus2_essentia = []
tristimulus3_essentia = []

#Basic Settings
##frame and hop size for spectral and harmonic features can be different
frameSize = 480
hopSize = 320
fs = 16000
frameSize_har = 1600
hopSize_har = 320
index = 0
specLength, harLength = 0, 0


#Load Sound
##Load audio data and sample rate with librosa
audio_librosa, fs_librosa = librosa.load(file_path, sr=None)

##Load audio data and sample rate with pyAudioAnalysis
[fs_pyAudio, audio_pyAudio] = audioBasicIO.readAudioFile(file_path);

##Load sound data with Essentia loader
loader = essentia.standard.MonoLoader(filename = file_path, sampleRate = fs)
audio = loader()

#Module Initialization for Essentia
##Define Essentia function with user preset
fft = FFT()
spectrum = Spectrum()
pitch = PitchYin(sampleRate = fs, frameSize = frameSize)
window = Windowing(type='hann')
energy = Energy()

specPeak = SpectralPeaks(minFrequency=20, maxFrequency = 8000, sampleRate = fs)
harPeak = HarmonicPeaks()

tristimulus = Tristimulus()
inharmonicity = Inharmonicity()


#Librosa feature extraction
specCentroid_librosa = librosa.feature.spectral_centroid(
        y=audio_librosa, sr=fs_librosa, n_fft=frameSize, hop_length=hopSize+1, freq=None)[0]
specFlatness_librosa = librosa.feature.spectral_flatness(
        y=audio_librosa, S=None, n_fft=frameSize, hop_length=hopSize+1)[0]
##Calculate Energy based on data loaded with librosa
energy_librosa = np.array([
    sum(abs(audio_librosa[i:i+frameSize]**2))
    for i in range(0, len(audio_librosa), hopSize+1)
])

#pyAudioAnalysis feature extraction
##Array F consists of multiple extracted features that pyAudioAnalysis supports
##Array f_names consists of name of extracted features
##Spectral spread feature's index is 4
F, f_names = audioFeatureExtraction.stFeatureExtraction(audio_pyAudio, fs_pyAudio, frameSize, hopSize-1);
specSpread_pyAudio = F[4,:]


#Extractin spectral features using Essentia
##FrameGenerator generates frame in audio data based on pre-defined frame and hop size
for frame in FrameGenerator(audio, frameSize = frameSize, hopSize = hopSize, startFromZero = True):
    ##Extract spectral slope feature using pre-defined user function in specSlope.py
    ampSignal = calculateAmpSignal(window(frame), frameSize)
    specSlope_essentia.append(calculateSpectralSlope(ampSignal, fs, frameSize))

#Extracting harmonic features using Essentia
for frame in FrameGenerator(audio, frameSize = frameSize_har, hopSize = hopSize_har, startFromZero = True):        
    #Pitch function returns pitch data(pitch_out) and pitch_confidence
    #If pitch_confidence equal to zero, it means that pitch data is not found in current frame
    pitch_out, pitch_confidence = pitch(frame)
    if pitch_confidence==0:
        pitch_out = 0
    
    #Spectrum of current frame is used to extract spectral peak feature
    #Spectral frequency and spectral magnitude will be returned
    specfrequency, specmagnitude = specPeak(spectrum(frame))
    
    #Spectral frequency, spectral magnitude, and pitch data are used to extract harmonic peak feature
    #Harmonic frequency and harmonic magnitude will be returned
    harfrequency, harmagnitude = harPeak(specfrequency, specmagnitude, pitch_out)
    
    #Calculate harmonic energy using harmonic frequency and harmonic magnitude
    harEnergySum = 0
    for i in range(len(harfrequency)):
        harEnergySum += harmagnitude[i] ** 2
    harEnergy_essentia.append(harEnergySum)
    
    #Total Energy(Spectral Energy) = Harmonic Energy + Noise Energy
    #Noisiness = Noise Energy/Total Energy
    totEnergy = energy(spectrum(frame))
    noiseEnergy_essentia.append(totEnergy-harEnergySum)
    noisiness_essentia.append((totEnergy-harEnergySum)/totEnergy)
    
    #Harmonic frequency and harmonic magnitude are used to extract inharmonicity and tristimulus feature
    #Tristimulus feature consists of 3 values per frame
    inharmonicity_essentia.append(inharmonicity(harfrequency, harmagnitude))
    tristimulus1_essentia.append(tristimulus(harfrequency, harmagnitude)[0])
    tristimulus2_essentia.append(tristimulus(harfrequency, harmagnitude)[1])
    tristimulus3_essentia.append(tristimulus(harfrequency, harmagnitude)[2])

#Length Variables
specLength = len(specSlope_essentia)
harLength = len(inharmonicity_essentia)

#Data Combine
spectral_column = ["Centroid", "Spread", "Energy", "FLatness", "Slope"]
harmonic_column = ["Inharmonicity", "Tristimulus1", "Tristimulus2", 
                   "Tristimulus3", "Harmonic Energy", "Noise Energy", "Noisiness"]
spectral_pd = pd.DataFrame(columns = spectral_column)
harmonic_pd = pd.DataFrame(columns = harmonic_column)

#Combining Spectral Data using Pandas
while index < specLength-1: 
    spectral_pd.loc[index] = [
            specCentroid_librosa[index], specSpread_pyAudio[index], 
            energy_librosa[index], specFlatness_librosa[index], specSlope_essentia[index]
            ]
    index += 1
index = 0


#Combining Harmonic Data using Pandas
while index < harLength: 
    harmonic_pd.loc[index] = [
            inharmonicity_essentia[index], tristimulus1_essentia[index], 
            tristimulus2_essentia[index], tristimulus3_essentia[index], 
            harEnergy_essentia[index], noiseEnergy_essentia[index],
            noisiness_essentia[index]
            ]
    index += 1

    
#Pandas to csv
spectral_pd.to_csv(spectral_path, index=False)
harmonic_pd.to_csv(harmonic_path, index=False)
