#!/usr/bin/env python 
"""Python script for downloading and  preprocessing German-credit data."""

import numpy as np
import urllib.request

# load file object from UCI data repository URL
text_file = urllib.request.urlopen(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/'
    'statlog/german/german.data-numeric')

# load data file to numpy array
raw_data = np.loadtxt(text_file)

# separate input feature columns (all but final)
inputs = raw_data[:, :-1]

# normalise input features to zero-mean, unit-variance
inputs_norm = (inputs - inputs.mean(0)) / inputs.std(0)

# add column of 1s for intercept coefficient
inputs_norm = np.concatenate(
    [inputs_norm, np.ones((inputs_norm.shape[0],1))], axis=1)
    
# output labels are in last column
outputs = raw_data[:, -1]

# recode output labels from {1,2} -> {0,1} values
outputs[outputs == 1] = 0
outputs[outputs == 2] = 1

# save arrays to text files
np.savetxt('R.txt', inputs_norm)
np.savetxt('S.txt', outputs)

