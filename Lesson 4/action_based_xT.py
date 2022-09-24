# Lesson 4: Expected Threat - Action-based
# https://soccermatics.readthedocs.io/en/latest/lesson4/xTAction.html
#%%
import pandas as pd
import json
import os
import pathlib
import warnings
import joblib
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations_with_replacement
# modelling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor, XGBClassifier


## Load data ##
# this was created in the possession_chains.py, with identification possession chains etc.
df = pd.DataFrame()
for i in range(11):
    file_name = 'possession_chains_England' + str(i+1) + '.json'
    path = os.path.join(str(pathlib.Path().resolve().parents[0]), "data", "possession_chain", file_name)
    with open(path) as f:
        data = json.load(f)
    df = pd.concat([df, pd.DataFrame(data)])
df = df.reset_index()
#%%
## Prepare variables for the models ##
# we use non-linear combinations of coordinates and distance to halfway line (c)
# model variables
var = ["x0", "x1", "c0", "c1"]
# combinations
inputs = []
# one variable combinations
inputs.extend(combinations_with_replacement(var, 1))
# 2 variable combinations
inputs.extend(combinations_with_replacement(var, 2))
# 3 variable combinations
inputs.extend(combinations_with_replacement(var, 3))

# make new columns
for i in inputs:
    # columns length 1 already exist
    if len(i) > 1:
        # column name
        column = ''
        x = 1
        for c in i:
            # add column name to be x0x1c0 for example
            column += c
            # multiply values in column
            x = x*df[c]
        # create a new column in df
        df[column] = x
        # add column to model variables
        var.append(column)
# investigate 3 columns
# df[var[-3:]].head(3)

## Calculate xT values for passes ##
### TRAINING, it's not perfect ML procedure, but results in AUC 0.2 higher than Logistic Regression ###
# passes = df.loc[ df["eventName"].isin(["Pass"])]
# X = passes[var].values #note that this is different X, with data from BL
# y = passes["shot_end"].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 123, stratify = y)
# xgb = XGBClassifier(n_estimators = 100, ccp_alpha=0, max_depth=4, min_samples_leaf=10,
#                    random_state=123)

# scores = cross_val_score(estimator = xgb, X = X_train, y = y_train, cv = 10, n_jobs = -1)
# print(np.mean(scores), np.std(scores))
# xgb.fit(X_train, y_train)
# print(xgb.score(X_train, y_train))
# y_pred = xgb.predict(X_test)
# print(xgb.score(X_test, y_test))

# save model
# path_model = os.path.join(str(pathlib.Path().resolve().parents[0]), 'possession_chain', 'finalized_model.sav')
path_model = "../data/possession_chain/finalized_model.sav"
# joblib.dump(xgb, path_model)
#%%
# predict if ended with shot
passes = df.loc[df["eventName"].isin(["Pass"])]
X = passes[var].values
y = passes["shot_end"].values

# load fitted model
model = joblib.load(path_model)
# predict probability of shot ended
y_pred_proba = model.predict_proba(X)[::,1]

passes["shot_prob"] = y_pred_proba
# OLS
shot_ended = passes.loc[passes["shot_end"] == 1]
X2 = shot_ended[var].values
y2 = shot_ended["xG"].values
lr = LinearRegression()
lr.fit(X2, y2)
y_pred = lr.predict(X2)
shot_ended["xG_pred"] = y_pred
# calculate xGchain
shot_ended["xT"] = shot_ended["xG_pred"]*shot_ended["shot_prob"]

shot_ended[["xG_pred", "shot_prob", "xT"]].head(5)

## Plot chain with passes and xT value ##
# extract chain
chain = df.loc[df["possesion_chain"] == 4]
# get passes
passes_in = shot_ended.loc[df["possesion_chain"] == 4]
max_value = passes_in["xT"].max()
# get events different than pass
not_pass = chain.loc[chain["eventName"] != "Pass"].iloc[:-1]
# shot is the last event of the chain (or should be)
shot = chain.iloc[-1]
# plot
pitch = Pitch(line_color='black',pitch_type='custom', pitch_length=105, pitch_width=68, line_zorder = 2)
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
# add size adjusted arrows
for i, row in passes_in.iterrows():
    value = row["xT"]
    # adjust the line width so that the more passes, the wider the line
    line_width = (value / max_value * 10)
    # get angle
    angle = np.arctan((row.y1-row.y0)/(row.x1-row.x0))*180/np.pi
    # plot lines on the pitch
    pitch.arrows(row.x0, row.y0, row.x1, row.y1,
                        alpha=0.6, width=line_width, zorder=2, color="blue", ax = ax["pitch"])
    # annotate text
    ax["pitch"].text((row.x0+row.x1-8)/2, (row.y0+row.y1-4)/2, str(value)[:5], fontweight = "bold", color = "blue", zorder = 4, fontsize = 20, rotation = int(angle))

# shot
pitch.arrows(shot.x0, shot.y0,
            shot.x1, shot.y1, width=line_width, color = "red", ax=ax['pitch'], zorder =  3)
# other passes like arrows
pitch.lines(not_pass.x0, not_pass.y0, not_pass.x1, not_pass.y1, color = "grey", lw = 1.5, ls = 'dotted', ax=ax['pitch'])
ax['title'].text(0.5, 0.5, 'Passes leading to a shot', ha='center', va='center', fontsize=30)


#%%
## Identify players with high xT ##
summary = shot_ended[["playerId", "xT"]].groupby(["playerId"]).sum().reset_index()
# add player name
player_path = "../data/Wyscout/players.json"
player_df = pd.read_json(player_path, encoding='unicode-escape')
player_df.rename(columns = {'wyId': 'playerId'}, inplace=True)
player_df["role"] = player_df.apply(lambda x: x.role["name"], axis = 1)
to_merge = player_df[['playerId', 'shortName', 'role']]

summary = summary.merge(to_merge, how = "left", on = ["playerId"])
#%%
# get minutes
minutes_path = "../data/minutes_played/minutes_played_per_game_England.json"
# filtering over 400 per game
minutes_per_game = pd.read_json(minutes_path, encoding="unicode-escape")
minutes = minutes_per_game.groupby(["playerId"]).sum().reset_index()
summary = minutes.merge(summary, how = "left", on = ["playerId"])
summary = summary.fillna(0)
summary = summary.loc[summary["minutesPlayed"] > 400]
# calculating per 90
summary["xT_p90"] = summary["xT"]*90/summary["minutesPlayed"]
#%%
# adjusting for possesion
player_poss_path = "../data/minutes_played/player_possesion_England.json"
percentage_df = pd.read_json(player_poss_path)
# merge
summary = summary.merge(percentage_df, how = "left", on = ["playerId"])
# adjust per possesion
summary["xT_adjusted_per_90"] = (summary["xT"]/summary["possesion"])*90/summary["minutesPlayed"]
summary[['shortName', 'xT_adjusted_per_90']].sort_values(by='xT_adjusted_per_90', ascending=False).head(5)