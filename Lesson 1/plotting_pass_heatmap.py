#%%
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch, Sbopen, VerticalPitch

#open the data
parser = Sbopen()
df_match = parser.match(competition_id=72, season_id=30)
# choose team
team = "England Women's"
# get list of games by our team, either home or away
match_ids = df_match.loc[(df_match["home_team_name"] == team) | (df_match["away_team_name"] == team)]["match_id"].tolist()
# calculate number of games
no_games = len(match_ids)

#declare an empty dataframe
danger_passes = pd.DataFrame()
for idx in match_ids:
    #open the event data from this game
    df = parser.event(idx)[0]
    for period in [1, 2]:
        
        # Get passes
        #keep only accurate passes by England that were not set pieces in this period
        mask_pass = ((df.team_name == team) &
                     (df.type_name == "Pass") &
                     (df.outcome_name.isnull()) &
                     (df.period == period) &
                     (df.sub_type_name.isnull()))
        passes = df.loc[mask_pass, ["x", "y", "end_x", "end_y", "minute", "second", "player_name"]]
        
        # Get shots
        #keep only Shots by England in this period
        mask_shot = ((df.team_name == team) &
                     (df.type_name == "Shot") &
                     (df.period == period))
        shots = df.loc[mask_shot, ["minute", "second"]]
        # convert time to seconds
        shot_times = shots['minute']*60 + shots['second']
        # find starts of the window
        shot_window = 15
        shot_start = shot_times - shot_window
        # condition to avoid negative shot starts
        shot_start = shot_start.apply(lambda x: x if x > 0 else (period-1)*45)
        # convert to seconds
        pass_times = passes['minute']*60 + passes['second']
        # check if pass is in any of the windows for this half
        pass_to_shot = pass_times.apply(lambda x: True in ((shot_start < x) & (x < shot_times)).unique())

        # keep only danger passes (15 sec. leading up to shot)
        danger_passes_period = passes.loc[pass_to_shot]
        #concatenate dataframe with a previous one to keep danger passes from the whole tournament
        danger_passes = pd.concat([danger_passes, danger_passes_period])

## ----------- PLOT LOCATIONS OF DANGER PASSES ------------- ##
# plot pitch
pitch = Pitch(line_color='black')
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
# plot passes
pitch.lines(danger_passes.x, danger_passes.y,
            danger_passes.end_x, danger_passes.end_y, comet=True,
            ax=ax["pitch"], lw=6, color="b", alpha=0.4)
pitch.scatter(danger_passes.end_x, danger_passes.end_y, ax=ax["pitch"],
              s=200, edgecolor="w", lw=3, color="b", alpha=1, zorder=5)

## ---------- PLOT HEATMAP OF DANGER PASSES ------------ ##
#plot pitch
pitch = Pitch(line_zorder=2, line_color='black')
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
# get the 2D histogram
bin_statistic = pitch.bin_statistic(danger_passes.x, danger_passes.y,
                                    statistic='count', bins=(6, 5), normalize=False)
# normalize by number of games
bin_statistic["statistic"] = bin_statistic["statistic"] / no_games
# make a heatmap
pcm  = pitch.heatmap(bin_statistic, cmap='Reds', edgecolor='grey', ax=ax['pitch'])
pitch.scatter(danger_passes.x, danger_passes.y, ax=ax["pitch"],
              s=200, color="k", alpha=0.2, zorder=1)
# legend to our plot
ax_cbar = fig.add_axes((1, 0.093, 0.03, 0.786))
cbar = plt.colorbar(pcm, cax=ax_cbar)

#%%
## ---------- CHALLENGE ------------ ##
# 1. Improve so that only high xG (>0.07) are included!
#declare an empty dataframe
danger_passes_high_xg = pd.DataFrame()
for idx in match_ids:
    #open the event data from this game
    df = parser.event(idx)[0]
    for period in [1, 2]:
        
        # Get passes
        #keep only accurate passes by England that were not set pieces in this period
        mask_pass = ((df.team_name == team) &
                     (df.type_name == "Pass") &
                     (df.outcome_name.isnull()) &
                     (df.period == period) &
                     (df.sub_type_name.isnull()))
        passes = df.loc[mask_pass, ["x", "y", "end_x", "end_y", "minute", "second", "player_name"]]
        
        # Get shots
        #keep only Shots by England in this period and with high xG
        mask_shot = ((df.team_name == team) &
                     (df.type_name == "Shot") &
                     (df.period == period) &
                     (df.shot_statsbomb_xg >= 0.07))
        shots = df.loc[mask_shot, ["minute", "second"]]
        # convert time to seconds
        shot_times = shots['minute']*60 + shots['second']
        # find starts of the window
        shot_window = 15
        shot_start = shot_times - shot_window
        # condition to avoid negative shot starts
        shot_start = shot_start.apply(lambda x: x if x > 0 else (period-1)*45)
        # convert to seconds
        pass_times = passes['minute']*60 + passes['second']
        # check if pass is in any of the windows for this half
        pass_to_shot = pass_times.apply(lambda x: True in ((shot_start < x) & (x < shot_times)).unique())

        # keep only danger passes (15 sec. leading up to shot)
        danger_passes_period = passes.loc[pass_to_shot]
        #concatenate dataframe with a previous one to keep danger passes from the whole tournament
        danger_passes_high_xg = pd.concat([danger_passes_high_xg, danger_passes_period])
        
#plot pitch
pitch = Pitch(line_zorder=2, line_color='black')
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
# get the 2D histogram
bin_statistic = pitch.bin_statistic(danger_passes_high_xg.x, danger_passes_high_xg.y,
                                    statistic='count', bins=(6, 5), normalize=False)
# normalize by number of games
bin_statistic["statistic"] = bin_statistic["statistic"] / no_games
# make a heatmap
pcm  = pitch.heatmap(bin_statistic, cmap='Reds', edgecolor='grey', ax=ax['pitch'])
pitch.scatter(danger_passes_high_xg.x, danger_passes_high_xg.y, ax=ax["pitch"],
              s=200, color="k", alpha=0.2, zorder=1)

#%%
# 2. Make a heat map only for Sweden’s player who was the most involved in danger passes!
# choose team
team = "Sweden Women's"
# get list of games by our team, either home or away
match_ids = df_match.loc[(df_match["home_team_name"] == team) | (df_match["away_team_name"] == team)]["match_id"].tolist()
# calculate number of games
no_games = len(match_ids)

#declare an empty dataframe
danger_passes_sweden = pd.DataFrame()
for idx in match_ids:
    #open the event data from this game
    df = parser.event(idx)[0]
    for period in [1, 2]:
        
        # Get passes
        #keep only accurate passes by England that were not set pieces in this period
        mask_pass = ((df.team_name == team) &
                     (df.type_name == "Pass") &
                     (df.outcome_name.isnull()) &
                     (df.period == period) &
                     (df.sub_type_name.isnull()))
        passes = df.loc[mask_pass, ["x", "y", "end_x", "end_y", "minute", "second", "player_name"]]
        
        # Get shots
        #keep only Shots by England in this period and with high xG
        mask_shot = ((df.team_name == team) &
                     (df.type_name == "Shot") &
                     (df.period == period))
        shots = df.loc[mask_shot, ["minute", "second"]]
        # convert time to seconds
        shot_times = shots['minute']*60 + shots['second']
        # find starts of the window
        shot_window = 15
        shot_start = shot_times - shot_window
        # condition to avoid negative shot starts
        shot_start = shot_start.apply(lambda x: x if x > 0 else (period-1)*45)
        # convert to seconds
        pass_times = passes['minute']*60 + passes['second']
        # check if pass is in any of the windows for this half
        pass_to_shot = pass_times.apply(lambda x: True in ((shot_start < x) & (x < shot_times)).unique())

        # keep only danger passes (15 sec. leading up to shot)
        danger_passes_period = passes.loc[pass_to_shot]
        #concatenate dataframe with a previous one to keep danger passes from the whole tournament
        danger_passes_sweden = pd.concat([danger_passes_sweden, danger_passes_period])

# filter for the sweden player with most danger passes
danger_player = danger_passes_sweden["player_name"].mode()[0]
danger_passes_player = danger_passes_sweden.loc[danger_passes_sweden["player_name"]==danger_player]

#number of games 
no_games = 0
for idx in match_ids:
    if danger_player in parser.event(idx)[0]["player_name"].unique():
        no_games += 1

#plot pitch
pitch = Pitch(line_zorder=2, line_color='black')
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
# get the 2D histogram
bin_statistic = pitch.bin_statistic(danger_passes_player.x, danger_passes_player.y,
                                    statistic='count', bins=(6, 5), normalize=False)
# normalize by number of games
bin_statistic["statistic"] = bin_statistic["statistic"] / no_games
# make a heatmap
pcm  = pitch.heatmap(bin_statistic, cmap='Reds', edgecolor='grey', ax=ax['pitch'])
pitch.scatter(danger_passes_player.x, danger_passes_player.y, ax=ax["pitch"],
              s=200, color="k", alpha=0.2, zorder=1)
