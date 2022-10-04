# 2. Assignment
# Christian Rønsholt
#####################################################

# Data can be stored in the same folder as the script
# The files required to run the script are:
# 1. wyscout event data from all 5 leagues (events_England.json etc.)
# 2. wyscout player data -> players.json
# 3. minutes played per game for all 5 leagues (minutes_played_per_game_England etc.)

# The code to add logos to the radars are commented out as they require the logos to be downloaded
# This also applies to the specific font used

#####################################################
#%%
import json
import numpy as np
import pandas as pd
from scipy import stats
# plotting
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from mplsoccer import PyPizza, add_image

# Load event data
df_eng = pd.read_json("events_England.json",
                     encoding = "unicode-escape")
df_ita = pd.read_json("events_Italy.json",
                     encoding = "unicode-escape")
df_fra = pd.read_json("events_France.json",
                     encoding = "unicode-escape")
df_ger = pd.read_json("events_Germany.json",
                     encoding = "unicode-escape")
df_spa = pd.read_json("events_Spain.json",
                     encoding = "unicode-escape")
df_pl = pd.concat([df_eng, df_ita, df_fra, df_ger, df_spa])

# Load player data
df_players = pd.read_json("players.json",
                          encoding="unicode-escape")

# Load minute data
with open("minutes_played_per_game_England.json") as f:
    minutes_per_game_england = json.load(f)
df_minutes_per_game_england = pd.DataFrame(minutes_per_game_england)
with open("minutes_played_per_game_France.json") as f:
    minutes_per_game_france = json.load(f)
df_minutes_per_game_france = pd.DataFrame(minutes_per_game_france)
with open("minutes_played_per_game_Germany.json") as f:
    minutes_per_game_germany = json.load(f)
df_minutes_per_game_germany = pd.DataFrame(minutes_per_game_germany)
with open("minutes_played_per_game_Italy.json") as f:
    minutes_per_game_italy = json.load(f)
df_minutes_per_game_italy = pd.DataFrame(minutes_per_game_italy)
with open("minutes_played_per_game_Spain.json") as f:
    minutes_per_game_spain = json.load(f)
df_minutes_per_game_spain = pd.DataFrame(minutes_per_game_spain)

df_minutes_per_game = pd.concat([df_minutes_per_game_england, df_minutes_per_game_france,
                                 df_minutes_per_game_germany, df_minutes_per_game_italy,
                                 df_minutes_per_game_spain])

df_minutes = df_minutes_per_game.groupby(["playerId"])["minutesPlayed"].sum().reset_index()

# clean event dataframe
df_pl["x"] = df_pl["positions"].apply(lambda row: (100 - row[0]["x"]) * 105/100)
df_pl["y"] = df_pl["positions"].apply(lambda row: row[0]["y"] * 68/100)

#%%
def calculate_ground_duels(df):
    ground_duels = df.loc[(df["subEventName"]=="Ground defending duel") &
                          (df["x"] < 52.5)]
    ground_duels_successful = ground_duels.loc[ground_duels.apply(lambda x: {"id": 1801} in x.tags, axis=1)]
    # sum by player
    ground_duels_player = ground_duels_successful.groupby("playerId")["eventId"].count().reset_index()
    ground_duels_player.rename(columns = {"eventId": "ground_duels"}, inplace=True)
    
    return ground_duels_player
    
def calculate_aerial_duels(df):
    aerial_duels = df.loc[(df["subEventName"]=="Air duel") &
                          (df["x"] < 52.5)]
    aerial_duels_successful = aerial_duels.loc[aerial_duels.apply(lambda x: {"id": 1801} in x.tags, axis=1)]
    # sum by player
    aerial_duels_player = aerial_duels_successful.groupby("playerId")["eventId"].count().reset_index()
    aerial_duels_player.rename(columns = {"eventId": "aerial_duels"}, inplace=True)
    
    return aerial_duels_player

def calculate_interceptions(df):
    interceptions = df.loc[df.apply(lambda x: {"id": 1401} in x.tags, axis=1)]
    interceptions_successful = interceptions.loc[interceptions.apply(lambda x: {"id": 1801} in x.tags, axis=1)]
    # sum by player
    interceptions_player = interceptions_successful.groupby("playerId")["eventId"].count().reset_index()
    interceptions_player.rename(columns = {"eventId": "interceptions"}, inplace=True)
    
    return interceptions_player

def calculate_passes(df):
    # get passes
    passes = df.loc[(df["eventName"]=="Pass")]
    # add end coordinates
    passes["end_x"] = passes["positions"].apply(lambda row: (100 - row[1]["x"]) * 105/100)
    passes["end_y"] = passes["positions"].apply(lambda row: row[1]["y"] * 68/100)
    
    # calculate passes from def to mid third
    passes_def_to_mid = passes.loc[(passes["x"] < 35) &
                                   (passes["end_x"].between(35, 70))]
    passes_def_to_mid_player = passes_def_to_mid.groupby("playerId")["eventId"].count().reset_index()
    passes_def_to_mid_player.rename(columns = {"eventId": "def_to_mid_passes"}, inplace=True)
    # calculate progressive passes
    # calculate distance to goal from pass origin
    passes["beginning"] = np.sqrt(np.square(105 - passes["x"]) + np.square(68 - passes["y"]))
    passes["end"] = np.sqrt(np.square(105 - passes["end_x"]) + np.square(68 - passes["end_y"]))
    passes.reset_index(inplace=True, drop=True)
    # get progressive passes (move the ball 25% closer to goal)
    passes["progressive"] = [(passes["end"][x]) / (passes["beginning"][x]) < .75 for x in range(len(passes["beginning"]))]
    # filter for progressive passes
    progressive_passes = passes.loc[passes["progressive"] == True].reset_index(drop=True)
    progressive_passes_player = progressive_passes.groupby("playerId")["eventId"].count().reset_index()
    progressive_passes_player.rename(columns = {"eventId": "progressive_passes"}, inplace=True)
    
    # merge progressive and def- to mid third passes
    df_passes = pd.merge(progressive_passes_player, passes_def_to_mid_player,
                         on="playerId")
    
    return df_passes

def calculate_true_tackles(df):
    total_tackles = df.loc[(df["subEventName"]=="Ground defending duel")]
    total_tackles_won = total_tackles.loc[total_tackles.apply(lambda x: {"id": 1801} in x.tags, axis=1)]
    total_tackles_lost = total_tackles.loc[total_tackles.apply(lambda x: {"id": 1802} in x.tags, axis=1)]
    fouls = df.loc[(df["subEventName"]=="Foul")]
    # remove not tackle fouls
    tackle_fouls = fouls.loc[fouls.apply(lambda x: {"id": 21} not in x.tags, axis=1)] #handball
    tackle_fouls = tackle_fouls.loc[tackle_fouls.apply(lambda x: {"id": 23} not in x.tags, axis=1)] #out of play foul
    tackle_fouls = tackle_fouls.loc[tackle_fouls.apply(lambda x: {"id": 24} not in x.tags, axis=1)] #protest
    tackle_fouls = tackle_fouls.loc[tackle_fouls.apply(lambda x: {"id": 26} not in x.tags, axis=1)] #time wasting
    tackle_fouls = tackle_fouls.loc[tackle_fouls.apply(lambda x: {"id": 27} not in x.tags, axis=1)] #dive

    # sum total tackles, tackles won and tackle fouls
    # tackle fouls
    tackle_fouls_player = tackle_fouls.groupby("playerId")["eventId"].count().reset_index()
    tackle_fouls_player.rename(columns = {"eventId": "tackle_fouls"}, inplace=True)
    # tackles won
    total_tackles_won_player = total_tackles_won.groupby("playerId")["eventId"].count().reset_index()
    total_tackles_won_player.rename(columns = {"eventId": "total_tackles_won"}, inplace=True)
    # tackles won
    total_tackles_lost_player = total_tackles_lost.groupby("playerId")["eventId"].count().reset_index()
    total_tackles_lost_player.rename(columns = {"eventId": "total_tackles_lost"}, inplace=True)

    # merge and calculate true tackle win rate
    df_true_tackles = (total_tackles_lost_player.merge(total_tackles_won_player, on="playerId")
                                                .merge(tackle_fouls_player, on="playerId"))
    df_true_tackles["true_tackle_win_rate"] = (df_true_tackles["total_tackles_won"] /
                                              (df_true_tackles["total_tackles_won"] +
                                               df_true_tackles["total_tackles_lost"] +
                                               df_true_tackles["tackle_fouls"]))
    df_true_tackles["true_tackles"] = (df_true_tackles["total_tackles_won"] +
                                       df_true_tackles["total_tackles_lost"] +
                                       df_true_tackles["tackle_fouls"])
    df_true_tackles["true_tackle_win_rate"] = df_true_tackles["true_tackle_win_rate"] * 100
    
    return df_true_tackles[["playerId", "true_tackles", "true_tackle_win_rate"]]

# Calculate ground duels, aerials and similar
df_passes = calculate_passes(df_pl)
df_aerial_duels = calculate_aerial_duels(df_pl)
df_interceptions = calculate_interceptions(df_pl)
df_true_tackles = calculate_true_tackles(df_pl)

# concatenate calculated metrics
players = df_pl["playerId"].unique()
df_summary = pd.DataFrame(players, columns = ["playerId"])
df_summary = (
    df_summary
    .merge(df_aerial_duels, how="left", on="playerId")
    .merge(df_interceptions, how="left", on="playerId")
    .merge(df_passes, how="left", on="playerId")
    .merge(df_true_tackles, how="left", on="playerId")
           )

# filter minutes
df_summary = df_minutes.merge(df_summary, how="left", on=["playerId"])
df_summary = df_summary.fillna(0)
df_summary = df_summary.loc[df_summary["minutesPlayed"] > 400]

# filter position
defenders = df_players.loc[df_players.apply(lambda x: x.role["name"] == "Defender", axis=1)]
defenders.rename(columns = {"wyId": "playerId"}, inplace=True)
to_merge = defenders[["playerId", "shortName"]]
df_summary = df_summary.merge(to_merge, how="inner", on=["playerId"])

# calculate per 90
summary_per_90 = pd.DataFrame()
summary_per_90["shortName"] = df_summary["shortName"]
for column in df_summary.columns[2:-2]:
    summary_per_90[column + "_per90"] = df_summary.apply(lambda x: x[column] * 90 / x["minutesPlayed"], axis=1)
# add columns not per 90
summary_per_90["true_tackle_win_rate"] = df_summary["true_tackle_win_rate"]

#%%
# Calculate percentiles
player = "S. Kolašinac"
team = "Arsenal FC"
df_player = summary_per_90.loc[summary_per_90["shortName"] == player]
df_player = df_player[["interceptions_per90", "aerial_duels_per90",
                       "progressive_passes_per90", "def_to_mid_passes_per90",
                       "true_tackles_per90", "true_tackle_win_rate"]]
per_90_columns = df_player.columns[:]
# values to mark on the plot
values = [round(df_player[column].iloc[0], 2) for column in per_90_columns]
# percentiles
percentiles = [int(stats.percentileofscore(summary_per_90[column], df_player[column].iloc[0])) for column in per_90_columns]


###############################################
                ## PLOT RADAR ##
###############################################
# metric names
names = ["Interceptions", "Air Duels Won",
         "Progressive Passes", "Passes from\nDef- to Mid Third",
         "True Tackles", "True Tackles\nWin Rate"]
arr1 = np.asarray(percentiles)
slice_colors = plt.cm.Reds(arr1 / 120)
text_colors = ["k"]*6
# set font
# mpl.rcParams["font.family"] = "Alegreya Sans"
# logo = Image.open(f"utils/{team}.png")

# Setup radar chart
baker = PyPizza(
    params=names,
    min_range = None, max_range = None,
    straight_line_color="k", straight_line_lw=1,
    last_circle_lw=5, other_circle_lw=2, other_circle_ls="--"
)
# create radar chart
fig, ax = baker.make_pizza(
    percentiles,
    figsize = (10, 10),
    param_location = 110, slice_colors=slice_colors,
    value_colors = text_colors, value_bck_colors=slice_colors,
    kwargs_slices = dict(edgecolor="k", zorder=2, linewidth=2, alpha=0.9),
    kwargs_params = dict(color="k", fontsize=16, va="center", fontweight="bold"),
    kwargs_values = dict(color="k", fontsize=16, zorder=3, fontweight="bold",
        bbox=dict(edgecolor="k", boxstyle="round,pad=0.3", lw=2))
)
# set per 90 values
texts = baker.get_value_texts()
for i, text in enumerate(texts):
    if i == len(texts)-1:
        text.set_text(f"{str(values[i])}%")
    else:
        text.set_text(str(values[i]))
# title
fig.text(0.515, 1, f"{player} - {team}",
         size=28, ha="center", color="k", fontweight="bold")
# subtitle
fig.text(0.515, 0.965, "Top 5 Leagues - 2017/18  |  Defenders  |  >400 min. played",
         size=15, ha="center", color="k")
# formatting
fig.set_facecolor("#eee9e5")
ax.set_facecolor("#eee9e5")
# # insert image
# ax_image = add_image(logo, fig,
#                      left=0.82, bottom=0.924,
#                      width=0.13, height=0.127)
# ax_image = add_image(logo, fig,
#                      left=0.11, bottom=0.924,
#                      width=0.13, height=0.127)

fig.savefig(f"radar_{player}.png", bbox_inches="tight", dpi=300)