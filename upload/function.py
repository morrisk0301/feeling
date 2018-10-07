#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 22:01:13 2018

@author: root
"""

import matplotlib.pyplot as plt
import numpy as np

def make_image(data, outputname):
    fig = plt.imshow(data)
    #fig.set_cmap('hot')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(outputname, bbox_inches = 'tight', pad_inches=0)
    
def resize_as_image(data):
    output = np.resize(data, (len(data), len(data)))
    return output
