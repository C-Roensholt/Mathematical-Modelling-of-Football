#%%
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch, Sbopen

# Load data
parser = Sbopen()
df, related, freeze, tactics = parser.event(69301)
# get passes
passes = df.loc[df['type_name'] == 'Pass'].loc[df['sub_type_name'] != 'Throw-in'].set_index('id')

## ------------- PLOT PLAYER PASSES ----------- ##
player = "Lucy Bronze"
mask_player = (df.type_name == 'Pass') & (df.player_name == player)
df_pass = df.loc[mask_player, ['x', 'y', 'end_x', 'end_y']]

pitch = Pitch(line_color='black')
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)

pitch.lines(df_pass.x, df_pass.y,
            df_pass.end_x, df_pass.end_y,
            color = "blue", ax=ax['pitch'], comet=True)
pitch.scatter(df_pass.end_x, df_pass.end_y, ax=ax['pitch'],
              alpha = 1, s = 500, color = "blue", edgecolor="w", lw=4,
              zorder=5)

fig.suptitle("Lucy Bronze passes against Sweden", fontsize = 30)

#%%
## ------------ MULTIPLE PASS PLOTS ------------- ##
#prepare the dataframe of passes by England that were no-throw ins
mask_england = (df.type_name == 'Pass') & (df.team_name == "England Women's") & (df.sub_type_name != "Throw-in")
df_passes = df.loc[mask_england, ['x', 'y', 'end_x', 'end_y', 'player_name']]
#get the list of all players who made a pass
names = df_passes['player_name'].unique()

#draw 4x4 pitches
pitch = Pitch(line_color='black', pad_top=20)
fig, axs = pitch.grid(ncols = 4, nrows = 4, grid_height=0.85, title_height=0.06, axis=False,
                      endnote_height=0.04, title_space=0.04, endnote_space=0.01)

#for each player
for name, ax in zip(names, axs['pitch'].flat[:len(names)]):
    #put player name over the plot
    ax.set_title(name,
                 ha='center', va='top', fontsize=14)
    #take only passes by this player
    player_df = df_passes.loc[df_passes["player_name"] == name]
    #plot arrow
    pitch.lines(player_df.x, player_df.y,
                 player_df.end_x, player_df.end_y, color = "blue",
                 ax=ax, comet=True, lw=1)
    #scatter
    pitch.scatter(player_df.end_x, player_df.end_y, alpha = 1,
                  s = 50, color = "blue", ax=ax, edgecolor="w", lw=2)

#We have more than enough pitches - remove them
for ax in axs['pitch'][-1, 16 - len(names):]:
    ax.remove()

#Another way to set title using mplsoccer
axs['title'].text(0.5, 0.5, 'England passes against Sweden', ha='center', va='center', fontsize=30)
