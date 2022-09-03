#%%
import numpy as np
import pandas as pd
from mplsoccer import Pitch, VerticalPitch, Sbopen
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

from utils.metadata import gradient, passing_columns, passing_names, radius

# choose team and player
team = "Spain"
player = "Pedro González López"

# Load StatsBomb data
parser = Sbopen()
df_competition = parser.match(competition_id=55, season_id=43)

# Load Fbref data
df_pass_types = pd.read_csv("data/pass_types.csv", skiprows=1)
df_passing = pd.read_csv("data/passing.csv")
df_fbref = pd.merge(df_pass_types, df_passing,
                    on=["Player", "Pos", "Squad", "Age", "Born"]).set_index("Player")[passing_columns]
# calculate percentile
df_fbref_percentile = df_fbref.rank(pct=True, axis=0, numeric_only=True)

df_fbref_player = df_fbref.loc[("Pedri")]
df_fbref_percentile_player = df_fbref_percentile.loc[("Pedri")]

# get list of games by our team, either home or away
match_ids = df_competition.loc[(df_competition["home_team_name"] == team) | (df_competition["away_team_name"] == team)]["match_id"].tolist()
# calculate number of games
no_games = len(match_ids)

# Load event data for all games
df_all_games = []
for id in match_ids:
    df_event, df_related, df_freeze, df_tactics = parser.event(id)
    df_all_games.append(df_event)
df_all_events = pd.concat(df_all_games)

# Filter for passes by the player (accurate and not set pieces)
mask_player = ((df_all_events.player_name == player) &
               (df_all_events.type_name == "Pass") &
               (df_all_events.outcome_name.isnull()) &
               (df_all_events.sub_type_name.isnull()))
df_player_passes = df_all_events.loc[mask_player]

# get different pass types
# CREATING CHANCES
# shot assist
df_player_shot_assist = df_player_passes[df_player_passes["pass_shot_assist"]==True]
# deep completions
def calculate_deep_completions(df):
    df['initialDistancefromgoal'] = np.sqrt(((120 - df['x'])**2) + ((40 - df['y'])**2))
    df['finalDistancefromgoal'] = np.sqrt(((120 - df['end_x'])**2) + ((40 - df['end_y'])**2))

    df['deepCompletion'] = (np.where(((df['finalDistancefromgoal'] <= (21.87)) &
                                      (df['initialDistancefromgoal'] >= (21.87))), True, False))
    df_deep_completion = df[df["deepCompletion"]==True]
    
    return df_deep_completion
df_deep_completions = calculate_deep_completions(df_player_passes)

# PROGRESSING FROM DEEP
# progressive passes (25% closer to goal)
def calculate_progressive_passes(df):
    df['beginning'] = np.sqrt(np.square(120 - df['x']) + np.square(40 - df['y']))
    df['end'] = np.sqrt(np.square(120 - df['end_x']) + np.square(40 - df['end_y']))
    df.reset_index(inplace=True, drop=True)
    # Get progressive passes
    df['progressive'] = [(df['end'][x]) / (df['beginning'][x]) < .75 for x in range(len(df['beginning']))]
    # Filter for progressive passes
    df = df.loc[df['progressive'] == True].reset_index(drop=True)
    return df
df_prog_passes = calculate_progressive_passes(df_player_passes)

# switches
df_switches = df_player_passes[df_player_passes["pass_switch"]==True]

# passes into final third
mask_final_third = (df_player_passes.end_x >= 80) & (df_player_passes.x <= 80)
df_final_third = df_player_passes.loc[mask_final_third]

# passes into penalty area
mask_penalty_area = ((df_player_passes.end_x >= 100) &
                     (df_player_passes.x <= 100) &
                     (df_player_passes.end_y.between(18, 62)))
df_penalty_area = df_player_passes.loc[mask_penalty_area]

## --------- PLOT PLAYER PASSES --------- ##
# create colormap
soc_cm = mcolors.LinearSegmentedColormap.from_list('SOC', gradient, N=50)
cm.register_cmap(name='SOC', cmap=soc_cm)
norm = mcolors.Normalize(vmin=0,
                         vmax=1)
cmap = plt.get_cmap('SOC')

# draw pitch
pitch = Pitch(line_color="black")
vert_pitch = VerticalPitch(line_color="black", half=True)

layout = [["A", "B", "C"]]
fig, axes = plt.subplot_mosaic(layout, constrained_layout=True, figsize=(20,12),
                               gridspec_kw={"width_ratios": [2, 1, 0.2]})

## -------- PLOT FINAL THIRD AND INTO PENALTY BOX PASSES (PITCH A)-------- ##
vert_pitch.draw(ax=axes["A"])
# get the 2D histogram
bin_statistic = vert_pitch.bin_statistic(df_prog_passes.x, df_prog_passes.y,
                                         statistic="count", bins=(6, 5), normalize=True)
# make a heatmap
pcm  = vert_pitch.heatmap(bin_statistic, cmap=cmap, edgecolor="k",
                          zorder=-2, ax=axes["A"], lw=3, ls="--")
vert_pitch.arrows(df_prog_passes.x, df_prog_passes.y,
                  df_prog_passes.end_x, df_prog_passes.end_y,
                  color="k", alpha=0.2, zorder=1,
                  headaxislength=4, headwidth=5, headlength=5,
                  ax=axes["A"])
vert_pitch.scatter(df_prog_passes.x, df_prog_passes.y, ax=axes["A"],
                   s=200, color="k", alpha=0.2, zorder=1)
labels = vert_pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=22,
                                  ax=axes["A"], ha='center', va='center',
                                  str_format='{:.0%}')
## ------- PLOT PERCENTILE RANKING (E) ----------- ##
bars = axes["B"].barh(y=df_fbref_percentile_player.index,
               width=df_fbref_percentile_player.values,
               edgecolor="k", hatch="///",
               color = cmap(norm(df_fbref_percentile_player.values)))
axes["B"].bar_label(bars, labels=df_fbref_player.values,
             fontweight="bold", fontsize=14)

# add vertical line
axes["B"].axvline(0.5, 0, 1, zorder=5, lw=2, color="k", ls="--")

# formatting
axes["B"].set_yticks(range(len(passing_names)), passing_names, fontsize=12, fontweight="bold")
axes["B"].set_xticks([0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1], fontsize=12)
axes["B"].spines.right.set_visible(False)
axes["B"].spines.top.set_visible(False)

# legend to our plot
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=axes["C"])