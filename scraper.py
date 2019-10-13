#%%

from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

def import_wine_quality(subset_size=0, verbose=False):
    # Open the  dataset as a pandas DataFrame
    if (verbose) : print("Open the  dataset as a pandas DataFrame")
    df = pd.read_csv("wine-quality/wine_white.csv", sep=';')

    # Sample the dataset
    if (verbose) : print("Sample the dataset")
    if (subset_size>0) : df = df[:subset_size]
    if (verbose) : print("    Size of the dataset : {}".format(len(df.index)))

    # Split the data
    print("Split the data")
    x = df.loc[:, df.columns != 'quality']
    y = df[['quality']]

    # Normalize X
    if (verbose) : print("Normalize X")
    min_max_scaler = preprocessing.MinMaxScaler()
    for tonorm in x.columns:
        temp = x[[tonorm]].values
        temp_scaled = min_max_scaler.fit_transform(temp)
        x[[tonorm]] = pd.DataFrame(temp_scaled)

    # Crop y
    remap = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 1,
        6: 2,
        7: 3,
        8: 4,
        9: 4,
        10: 5
    }
    y = y['quality'].map(remap)

    y = pd.get_dummies(y)

    return x,y


#%%