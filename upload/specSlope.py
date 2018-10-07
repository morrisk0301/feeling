#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 14:55:03 2018

@author: Kyeong In Kim
"""

import numpy as np
import math


def calculateAmpSignal(rawSignal, frameSize):
    ampSignal = np.zeros(shape = (int(frameSize/2),))
    fftSignal = np.fft.fft(rawSignal)
    
    for i in range(int(frameSize/2)):
        ampSignal[i] = math.sqrt(
                math.pow(fftSignal[i].real, 2) +
                math.pow(fftSignal[i].imag, 2)
                )
        
    return ampSignal

def calculateSpectralSlope(ampSignal, fs, frameSize):
    ampSum = 0
    freqSum = 0
    powFreqSum = 0
    ampFreqSum = 0
    freqs = np.zeros(shape = (len(ampSignal),))
    
    for i in range(len(ampSignal)):
        ampSum += ampSignal[i]
        curFreq = i * fs / frameSize
        freqs[i] = curFreq
        powFreqSum += curFreq * curFreq;
        freqSum += curFreq;
        ampFreqSum += curFreq * ampSignal[i];
    
    return -(len(ampSignal) * ampFreqSum - freqSum * ampSum) / (ampSum * (
        powFreqSum - math.pow(freqSum, 2)))
