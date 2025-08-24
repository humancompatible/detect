from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def plot_3D_histogram(xdata,ydata,xlabel=None,ylabel=None,zlabel=None):

    x = xdata   # turn x,y data into numpy arrays
    y = ydata

    fig = plt.figure(frameon='False')#create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')

    #make histogram stuff - set bins
    hist, xedges, yedges = np.histogram2d(x, y, bins=(5,3), density=True)
    xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])


    xpos = xpos.flatten()/2.
    ypos = ypos.flatten()/2.
    zpos = np.zeros_like (xpos)

    dx = .9*(xedges [1] - xedges [0])
    dy = .9*(yedges [1] - yedges [0])
    dz = hist.flatten()

    cmap = plt.cm.get_cmap('jet') # Get desired colormap - you can change this!
    max_height = np.max(dz)   # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz]

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average', )
    plt.title(zlabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def load_histograms(
    csv_path,
    target_col,
    protected_list,
    method_kwargs
    ):

    csv_path = Path(csv_path)

    df = pd.read_csv(csv_path)

    ############# Manual encoding #############
    df["Race_cod"] = df["Race"]
    df["Age_cod"] = df["Age"]

    for index, row in df.iterrows():
        if df["Race"][index] == 'Green':
            df["Race_cod"][index] = 1
        elif df["Race"][index] == 'Blue':
            df["Race_cod"][index] = 2
        elif df["Race"][index] == 'Purple':
            df["Race_cod"][index] = 3

        if df["Age"][index] == '0-18':
            df["Age_cod"][index] = 1
        elif df["Age"][index] == '18-30':
            df["Age_cod"][index] = 2
        elif df["Age"][index] == '30-45':
            df["Age_cod"][index] = 3
        if df["Age"][index] == '45-60':
            df["Age_cod"][index] = 4
        elif df["Age"][index] == '60+':
            df["Age_cod"][index] = 5
    ############# Manual encoding #############

    # Filter by positive target
    df = df[df[target_col] == 1]


    discr = df[df['Gender'] == 'F']
    priv = df[df['Gender'] == 'M']


    plot_3D_histogram(discr.loc[:, ('Age_cod')], discr.loc[:, ('Race_cod')],
                      xlabel="Age", ylabel="Race", zlabel="F")
    plot_3D_histogram(priv.loc[:, ('Age_cod')], priv.loc[:, ('Race_cod')],
                      xlabel="Age", ylabel="Race", zlabel="M")
