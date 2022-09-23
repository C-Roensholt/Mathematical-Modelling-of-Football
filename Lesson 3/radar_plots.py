# https://soccermatics.readthedocs.io/en/latest/gallery/lesson3/plot_RadarPlot.html
import pandas as pd
import numpy as np
import json
# plotting
import matplotlib.pyplot as plt
# statistical fitting of models
import statsmodels.api as sm
import statsmodels.formula.api as smf
#used for plots
from scipy import stats
from mplsoccer import PyPizza, FontManager

from utility_functions import calulatexG, FinalThird

# Load data
with open("../data/wyscout/events/events_England.json") as f:
    data = json.load(f)
train = pd.DataFrame(data)

# calculate xG
npxg = calulatexG(train, npxG = True)

# calculate final third entries
final_third = FinalThird(train)