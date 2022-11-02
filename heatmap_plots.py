# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 15:27:39 2022

@author: everall
"""

# ilona presenting first week of november
# 3 November
# %%
import sys
import operator
import numpy as np
import pandas as pd
import seaborn as sns
from statistics import stdev, median, mean
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os as os
import inspect
import itertools
import math
import re

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
sys.path.insert(0, currentdir)


# %% import data and joining together


