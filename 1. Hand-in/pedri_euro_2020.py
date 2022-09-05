#%%
import numpy as np
import pandas as pd
from mplsoccer import Pitch, VerticalPitch, Sbopen, PyPizza
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib import cm

from utils.metadata import (gradient,
                            passing_columns,
                            passing_names,
                            radius)
from utils.utility_functions import (calculate_deep_completions,
                                     calculate_progressive_passes)

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
for match_id in match_ids:
    df_event, df_related, df_freeze, df_tactics = parser.event(match_id)
    df_all_games.append(df_event)
df_all_events = pd.concat(df_all_games)

# Filter for passes by the player (accurate and not set pieces)
mask_player = ((df_all_events.player_name == player) &
               (df_all_events.type_name == "Pass") &
               (df_all_events.outcome_name.isnull()) &
               (df_all_events.sub_type_name.isnull()))
df_player_passes = df_all_events.loc[mask_player]

# CALCULATE DIFFERENT PASS TYPES
# shot assist
df_player_shot_assist = df_player_passes[df_player_passes["pass_shot_assist"]==True]
# deep completions
df_deep_completions = calculate_deep_completions(df_player_passes)
# progressive passes
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
cmap = plt.get_cmap('Reds')

# draw pitch
pitch = Pitch(line_color="black")
vert_pitch = VerticalPitch(line_color="black", half=True)

fig, ax = plt.subplots(figsize=(14,10))

## -------- PLOT FINAL THIRD AND INTO PENALTY BOX PASSES (PITCH A)-------- ##
vert_pitch.draw(ax=ax)
# get the 2D histogram
bin_statistic = vert_pitch.bin_statistic(df_prog_passes.x, df_prog_passes.y,
                                         statistic="count", bins=(6, 5), normalize=True)
# make a heatmap
pcm  = vert_pitch.heatmap(bin_statistic, cmap="Reds", edgecolor="k",
                          zorder=-2, ax=ax, lw=3, ls="--")

vert_pitch.scatter(df_prog_passes.x, df_prog_passes.y, ax=ax,
                   s=250, color="k", alpha=0.2, zorder=1)
labels = vert_pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=22,
                                  ax=ax, ha='center', va='center',
                                  str_format='{:.0%}')  

## ------- PLOT PERCENTILE RANKING (B) ----------- ##
mpl.rcParams['font.family'] = 'Alegreya Sans'

# setup variables
pvals = [x * 100 for x in df_fbref_percentile_player.values] #multiply to get 0-100 scale
arr1 = np.asarray(pvals)
N = len(pvals)
bottom = 0.0
theta, width = np.linspace(0.0, 2 * np.pi, N, endpoint=False, retstep=True)

# add ax for radar
ax_polar = fig.add_axes([0.9, 0.15, 0.8, 0.8], polar=True)
ax_polar.set_rorigin(-20)

# plot bars
bars = ax_polar.bar(
    theta, height=arr1, width=width,
    bottom=bottom, color="w", edgecolor="k", zorder=1, linewidth=4
)
# color different bars
for i, bar in enumerate(bars):
    bar.set_color(cmap(norm(df_fbref_percentile_player.values[i])))
    bar.set_edgecolor('k')

ax_polar.set_rticks(np.arange(0.0, 120.0, 20.0))
ax_polar.set_thetagrids((theta+width/2)* 180 / np.pi)
# axes["B"].set_rlabel_position(-100)
ax_polar.set_theta_zero_location("N")
ax_polar.set_theta_direction(-1)

# axes["B"].set_rticks([])
# ticks = [str(i+1) for i in range(len(pvals))]

strvals = [str(round(pvals[i])) for i in range(len(pvals))]
ax_polar.set_xticklabels([])
rotations = np.rad2deg(theta)

# Plot name and percentile labels
label_nums = []
for i, (x, bar, rotation, label, strlab) in enumerate(zip(theta, bars, rotations, passing_names, strvals)):
    if i==0 or i==2 or i==6:
        lab = ax_polar.text(x, 122, label,ha='center', va='center', color="k",
                            fontsize=20, fontweight='regular')
    elif i==11:
        lab = ax_polar.text(x, 132, label,ha='center', va='center', color="k",
                            fontsize=20, fontweight='regular')
    else:
        label_text = ax_polar.text(x, 128, label, ha='center', va='center_baseline', color="k",
                                   fontsize=20, fontweight='regular')
    
    # Add percentile numbers and adjust colors dependend on category
    label_number = ax_polar.text(x, bar.get_height(), strlab, ha='center', va='center', color='w',
                                 fontsize=20, zorder=5, fontweight='bold',
                                 bbox=dict(boxstyle='round', facecolor=cmap(norm(df_fbref_percentile_player.values[i])), alpha=1, edgecolor='k', linewidth=4))
    
    label_nums.append(label_number)
    
# Format spines
ax_polar.spines["polar"].set_color('k')
ax_polar.spines["polar"].set_linewidth(4)

# Grid inside plot
ax_polar.grid(b=True, axis='x', zorder=20, color='k', linewidth=4)
ax_polar.grid(b=True, axis='y', zorder=20, color='k', linestyle=(0, (5, 10)), linewidth=0.8)
ax_polar.spines['polar'].set_visible(True)
ax_polar.set_yticklabels([])

fig.tight_layout()

# legend to our plot
# ax_cbar = fig.add_axes([0.9, 0.15, 0.1, 0.8])
# cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap="Reds"), cax=ax_cbar)