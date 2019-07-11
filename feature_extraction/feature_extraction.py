# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 18:09:06 2018

@author: Kyeong In Kim
"""

import sys
import os
import librosa
import essentia
import essentia.streaming
import numpy as np
import pandas as pd
import glob
from essentia.standard import *
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from specSlope import *


# Error handling Function
def get_value(A, i, num):
    try:
        return A[i]
    except IndexError:
        return num


# Extracting Feature Function
def extract_feature(ori_file_name, folder_path):
    # make output dir
    if not os.path.exists('extract_result'):
        os.makedirs('extract_result')
    if not os.path.exists('extract_result/'+folder_path):
        os.makedirs('extract_result/'+folder_path)
        
    # file open
    file_name = os.path.splitext(os.path.basename(ori_file_name))[0]
    feature_path = "extract_result/"+folder_path+"/"+file_name+"_feature.csv"

    # Variable Declaration
    index = 0
    feature_length = 0
    specSlope_essentia = []
    pitch_essentia = []
    harEnergy_essentia = []
    noiseEnergy_essentia = []
    noisiness_essentia = []
    inharmonicity_essentia = []
    tristimulus1_essentia = []
    tristimulus2_essentia = []
    tristimulus3_essentia = []
    
    # Load Sound
    # Load audio data and sample rate with librosa
    audio_librosa, fs_librosa = librosa.load(ori_file_name, sr=None)
    
    # Load audio data and sample rate with pyAudioAnalysis
    [fs_pyAudio, audio_pyAudio] = audioBasicIO.readAudioFile(ori_file_name);
    
    # Load sound data with Essentia loader
    loader = essentia.standard.MonoLoader(filename = ori_file_name, sampleRate = fs_pyAudio)
    audio_essentia = loader()
    
    # Module Initialization for Essentia
    # Define Essentia function with user preset
    fft = FFT()
    spectrum = Spectrum()
    pitch = PitchYin(sampleRate = fs_pyAudio, frameSize = frameSize)
    window = Windowing(type='hann')
    energy = Energy()
    
    specPeak = SpectralPeaks(minFrequency=20, maxFrequency = 8000, sampleRate = fs_pyAudio)
    harPeak = HarmonicPeaks()
    
    tristimulus = Tristimulus()
    inharmonicity = Inharmonicity()
    
    # Librosa feature extraction
    specCentroid_librosa = librosa.feature.spectral_centroid(
            y=audio_librosa, sr=fs_librosa, n_fft=frameSize, hop_length=hopSize+1, freq=None)[0]
    specFlatness_librosa = librosa.feature.spectral_flatness(
            y=audio_librosa, S=None, n_fft=frameSize, hop_length=hopSize+1)[0]
    mfcc_librosa = librosa.feature.mfcc(
            y=audio_librosa, sr=fs_librosa, n_mfcc=16, n_fft=frameSize, hop_length=hopSize+1)
    de_mfcc_librosa = librosa.feature.delta(mfcc_librosa)
    # Calculate Energy based on data loaded with librosa
    energy_librosa = np.array([
        sum(abs(audio_librosa[i:i+frameSize]**2))
        for i in range(0, len(audio_librosa), hopSize+1)
    ])
    
    # pyAudioAnalysis feature extraction
    # Array F consists of multiple extracted features that pyAudioAnalysis supports
    # Array f_names consists of name of extracted features
    # Spectral spread feature's index is 4
    F, f_names = audioFeatureExtraction.stFeatureExtraction(audio_pyAudio, fs_pyAudio, frameSize, hopSize-1);
    specSpread_pyAudio = F[4,:]
    
    # Extractin spectral features using Essentia
    # FrameGenerator generates frame in audio data based on pre-defined frame and hop size
    for frame in FrameGenerator(audio_essentia, frameSize = frameSize, hopSize = hopSize, startFromZero = True):
        # Extract spectral slope feature using pre-defined user function in specSlope.py
        ampSignal = calculateAmpSignal(window(frame), frameSize)
        specSlope_essentia.append(calculateSpectralSlope(ampSignal, fs_pyAudio, frameSize))
    
    # Extracting harmonic features using Essentia
    for frame in FrameGenerator(audio_essentia, frameSize = frameSize_har, hopSize = hopSize_har, startFromZero = True):        
        # Pitch function returns pitch data(pitch_out) and pitch_confidence
        # If pitch_confidence equal to zero, it means that pitch data is not found in current frame
        pitch_out, pitch_confidence = pitch(frame)
        if pitch_confidence==0:
            pitch_out = 0
        pitch_essentia.append((pitch_out))

        # Spectrum of current frame is used to extract spectral peak feature
        # Spectral frequency and spectral magnitude will be returned
        specfrequency, specmagnitude = specPeak(spectrum(frame))
        
        # Spectral frequency, spectral magnitude, and pitch data are used to extract harmonic peak feature
        # Harmonic frequency and harmonic magnitude will be returned
        harfrequency, harmagnitude = harPeak(specfrequency, specmagnitude, pitch_out)
        
        # Calculate harmonic energy using harmonic frequency and harmonic magnitude
        harEnergySum = 0
        for i in range(len(harfrequency)):
            harEnergySum += harmagnitude[i] ** 2
        harEnergy_essentia.append(harEnergySum)
        
        # Total Energy(Spectral Energy) = Harmonic Energy + Noise Energy
        # Noisiness = Noise Energy/Total Energy
        totEnergy = energy(spectrum(frame))
        noiseEnergy_essentia.append(totEnergy-harEnergySum)
        try:
            noisiness = (totEnergy-harEnergySum)/totEnergy
        except ZeroDivisionError:
            noisiness = 0
        finally:
            noisiness_essentia.append(noisiness)
        
        # Harmonic frequency and harmonic magnitude are used to extract inharmonicity and tristimulus feature
        # Tristimulus feature consists of 3 values per frame
        inharmonicity_essentia.append(inharmonicity(harfrequency, harmagnitude))
        tristimulus1_essentia.append(tristimulus(harfrequency, harmagnitude)[0])
        tristimulus2_essentia.append(tristimulus(harfrequency, harmagnitude)[1])
        tristimulus3_essentia.append(tristimulus(harfrequency, harmagnitude)[2])
    
    # Length Variables
    feature_length = len(specSlope_essentia) if len(specSlope_essentia) > len(harEnergy_essentia) else len(harEnergy_essentia)
    
    # Data Combine
    spectral_column = ["MFCC_1", "MFCC_2", "MFCC_3", "MFCC_4", "MFCC_5", "MFCC_6",
                       "MFCC_7", "MFCC_8", "MFCC_9", "MFCC_10", "MFCC_11", "MFCC_12",
                       "MFCC_13", "MFCC_14", "MFCC_15", "MFCC_16", "de_MFCC_1", 
                       "de_MFCC_2", "de_MFCC_3", "de_MFCC_4", "de_MFCC_5", "de_MFCC_6", 
                       "de_MFCC_7", "de_MFCC_8", "de_MFCC_9", "de_MFCC_10", 
                       "de_MFCC_11", "de_MFCC_12", "de_MFCC_13", "de_MFCC_14", 
                       "de_MFCC_15", "de_MFCC_16", "Centroid", "Spread", "Energy", 
                       "Flatness", "Slope", "Pitch", "Inharmonicity", "Tristimulus1", "Tristimulus2",
                       "Tristimulus3", "Harmonic Energy", "Noise Energy", "Noisiness"]
    feature_pd = pd.DataFrame(columns = spectral_column)
    
    #Combining Spectral Data using Pandas
    while index < feature_length-1: 
        mfcc_list = [get_value(mfcc_librosa[i], index, 0) for i in range(16)]
        de_mfcc_list = [get_value(de_mfcc_librosa[i], index, 0) for i in range(16)]
        tot_list = mfcc_list + de_mfcc_list
        tot_list.extend((
                get_value(specCentroid_librosa, index, 0), 
                get_value(specSpread_pyAudio, index, 0),
                get_value(energy_librosa, index, 0),
                get_value(specFlatness_librosa, index, 0),
                get_value(specSlope_essentia, index, 0),
                get_value(pitch_essentia, index, 0),
                get_value(inharmonicity_essentia, index, 0),
                get_value(tristimulus1_essentia, index, 0),
                get_value(tristimulus2_essentia, index, 0),
                get_value(tristimulus3_essentia, index, 0),
                get_value(harEnergy_essentia, index, 0),
                get_value(noiseEnergy_essentia, index, 0),
                get_value(noisiness_essentia, index, 0)
                ))
        feature_pd.loc[index] = [tot_list[i] for i in range(44)]
        index += 1
        
    #Pandas to csv
    feature_pd.to_csv(feature_path, index=False)
    print("Extracted Features have been successfully saved!")
    

#Basic Settings
try:
    folder_path = sys.argv[1]
except IndexError as e:
    raise Exception('Invalid folder path!')
    #folder_path = 'test'
try:
    frameSize = sys.argv[2]
except IndexError as e:
    frameSize = 480
try:
    hopSize = sys.argv[3]
except IndexError as e:
    hopSize = 320
try:
    frameSize_har = sys.argv[4]
except IndexError as e:
    frameSize_har = 1600
try:
    hopSize_har = sys.argv[5]
except IndexError as e:
    hopSize_har = 320

sound_list = glob.glob(folder_path+'/*.wav')

for file in sound_list:
    extract_feature(file, folder_path)