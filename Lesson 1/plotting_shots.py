#%%
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch, Sbopen, VerticalPitch

# Load data
parser = Sbopen()
df, related, freeze, tactics = parser.event(69301)

# get team names
team1, team2 = df.team_name.unique()


## ----------- FIRST SHOT MAP --------- ##
# Plot pitch
pitch = Pitch(line_color='black')

fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)

# get team shots
mask_team1 = (df["type_name"]=="Shot") & (df["team_name"]==team1)
df_shots_team1 = df.loc[mask_team1, ["x","y", "outcome_name", "player_name", "shot_statsbomb_xg"]]
df_goals_team1 = df_shots_team1.loc[df_shots_team1["outcome_name"]=="Goal"]

mask_team2 = (df["type_name"]=="Shot") & (df["team_name"]==team2)
df_shots_team2 = df.loc[mask_team2, ["x","y", "outcome_name", "player_name", "shot_statsbomb_xg"]]
df_goals_team2 = df_shots_team2.loc[df_shots_team2["outcome_name"]=="Goal"]

# shots team1
pitch.scatter(df_shots_team1.x, df_shots_team1.y,
              alpha = 0.2, s = 500, color = "red", ax=ax['pitch'])
pitch.scatter(df_goals_team1.x, df_goals_team1.y,
              alpha = 1, s = 500, color = "red", ax=ax['pitch'])
# pitch.annotate(df_goals_team1["player_name"],
#                (df_goals_team1.x + 1, df_goals_team1.y - 2),
#                ax=ax['pitch'], fontsize = 12)

# shots team2
pitch.scatter(120-df_shots_team2.x, 80-df_shots_team2.y,
              alpha = 0.2, s = 500, color = "blue", ax=ax['pitch'])
pitch.scatter(120-df_goals_team2.x, 80-df_goals_team2.y,
              alpha = 1, s = 500, color = "blue", ax=ax['pitch'])


## ----------- VERTICAL SHOT MAP ----------- ##
pitch = VerticalPitch(line_color='black', half = True)
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)

#plotting all shots
for i, shot in df_shots_team1.iterrows():
    if shot["outcome_name"] == "Goal":
        pitch.scatter(shot.x, shot.y,
                      alpha = 1, s = shot["shot_statsbomb_xg"]*10000, color = "red",
                      ax=ax['pitch'], edgecolors="black")
    else:
        pitch.scatter(shot.x, shot.y,
                      alpha = 0.2, s = shot["shot_statsbomb_xg"]*10000, color = "red",
                      ax=ax['pitch'], edgecolors="black")
fig.suptitle("England shots against Sweden", fontsize = 30)
plt.show()

#%%
## ----------- CHALLENGE ---------- ##
# get sweden passes
mask_team2_pass = (df["type_name"]=="Pass") & (df["team_name"]==team2)
df_passes_team2 = df.loc[mask_team2_pass, :]

# get player passes
player = "Sara Caroline Seger"
df_player_passes = df_passes_team2.loc[df_passes_team2["player_name"]==player]

# plot passes
pitch = Pitch(line_color='black')

fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)

# plot all passes
pitch.arrows(xstart=df_passes_team2.x, ystart=df_passes_team2.y,
             xend=df_passes_team2.end_x, yend=df_passes_team2.end_y,
             ax=ax["pitch"], color="k", alpha=0.2,
             headlength=6, headwidth=8, headaxislength=3)
# plot player passes
pitch.arrows(xstart=df_player_passes.x, ystart=df_player_passes.y,
             xend=df_player_passes.end_x, yend=df_player_passes.end_y,
             ax=ax["pitch"], color="k", alpha=1,
             headlength=6, headwidth=8, headaxislength=3)
pitch.scatter(x=df_player_passes.x, y=df_player_passes.y,
              ax=ax["pitch"], s=400, color="k", edgecolor="w", lw=3)

fig.suptitle(f"Passes by {player} in {team1} vs. {team2}", fontsize = 24)