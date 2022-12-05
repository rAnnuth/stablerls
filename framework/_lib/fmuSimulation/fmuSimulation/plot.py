# Created by Robert Annuth
# robert.annuth@tuhh.de

import matplotlib as mpl
#from matplotlib.backends.backend_pgf import FigureCanvasPgf
#mpl.backend_bases.register_backend('pgf', FigureCanvasPgf)
import matplotlib.pyplot as plt
from math import sqrt

default_width = 6.3 # width in inches
default_ratio = (sqrt(5.0) - 1.0) / 3.0

#plt.rcParams.update({
#    "pgf.texsystem": "xelatex",
#    "font.family": "serif",
#    "text.usetex": True, # use inline math for ticks
#    "font.serif": [],
#    "font.cursive": [ "Comic Neue", "Comic Sans MS"],
#    "figure.figsize": [default_width, default_width * default_ratio],
#    "pgf.preamble": "\n".join([
#        # package and macros definitions are also possible, e.g.:
#        # has to be pdflatex and xelatex compatible
#    ]),
#})


"""
Wrapper function to return correctly sized figure
"""
def figure(width=default_width, ratio=default_ratio, pad=0, pad_left=0,
            pad_right=1, pad_top=1, pad_bot=0, *args, **kwargs):
    fig = plt.figure(figsize=(width, width * ratio), *args, **kwargs)
    fig.set_tight_layout({
        'pad': pad,
        'rect': [pad_left, pad_bot, pad_right, pad_top]
    })
    return fig


"""
Wrapper function to return correctly sized subplot figure
"""
def subplots(nrows=1, ncols=1, width=default_width, ratio=default_ratio, pad_left=0,
            pad_right=1, pad_top=1, pad_bot=0, *args, **kwargs):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, width * ratio), *args, **kwargs)
    fig.set_tight_layout({
        'pad': 0,
        'rect': [pad_left, pad_bot, pad_right, pad_top]
    })
    return fig, axes


"""
Wrapper function to save file with specified backend
"""
def savefig(filename, *args, **kwargs):
    plt.savefig(filename, *args, **kwargs) #'.pgf' or '.pdf'

