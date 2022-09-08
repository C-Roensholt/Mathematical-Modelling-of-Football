#%%
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from highlight_text import ax_text, fig_text
from mplsoccer import Pitch, VerticalPitch, Sbopen


from utils.metadata import (gradient,
                            passing_columns,
                            passing_names,
                            radius)
from utils.utility_functions import (calculate_deep_completions,
                                     calculate_progressive_passes,
                                     draw_arrow_with_shrink)

# choose team and player
team = "Spain"
player = "Pedro González López"

# Load StatsBomb data
parser = Sbopen()
df_competition = parser.match(competition_id=55, season_id=43)

# get list of games by our team, either home or away
match_ids = df_competition.loc[(df_competition["home_team_name"] == team) | (df_competition["away_team_name"] == team)]["match_id"].tolist()
# calculate number of games
no_games = len(match_ids)

# Load event data for all games
df_all_games = []
for match_id in match_ids:
    df_event, df_related, df_freeze, df_tactics = parser.event(match_id)
    df_all_games.append(df_event)
df_all_events = pd.concat(df_all_games).reset_index()

# Filter for passes by the player (accurate and not set pieces)
mask_player = ((df_all_events.player_name == player) &
               (df_all_events.type_name == "Pass") &
               (df_all_events.outcome_name.isnull()) &
               (df_all_events.sub_type_name.isnull()))
df_player_passes = df_all_events.loc[mask_player]

# CALCULATE DIFFERENT PASS TYPES
# shot assist
df_player_shot_assist = df_player_passes[df_player_passes["pass_shot_assist"]==True]
# merge to get shot info
df_shot_assist = df_player_shot_assist.merge(df_all_events,
                                             left_on="id", right_on="shot_key_pass_id",
                                             how="left", suffixes=["", "_shot"])
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

# CALCULATE TOTAL xG FOR SPAIN AND xCHAIN FOR PEDRI
mask_shot = ((df_all_events.team_name == team) &
             (df_all_events.type_name == "Shot") &
             (df_all_events.sub_type_name != "Penalty") &
             (df_all_events.sub_type_name != "Penalty Saved") &
             (df_all_events.sub_type_name == "Open Play"))
df_all_shots = df_all_events.loc[mask_shot]
shots_idxs = list(df_all_shots.index)
# create groups based on team sequences of possession
df_all_events['team_name_1'] = df_all_events['team_name'].shift(1)
df_all_events['team_mask'] = df_all_events['team_name'] != df_all_events['team_name_1']
df_all_events['team_groups'] = df_all_events['team_mask'].cumsum()

# collect xg in sequences for all players
xg_chain_dict = {}
for idx in shots_idxs:
    #access the shot sequence/group
    shot_sequence_group = df_all_events.loc[idx, :]['team_groups']
    df_shot_sequence = df_all_events[df_all_events['team_groups'] == shot_sequence_group]
    
    for player in list(df_shot_sequence["player_name"]):
        if player not in xg_chain_dict:
            xg_chain_dict[player] = df_shot_sequence["shot_statsbomb_xg"].sum()
        else:
            xg_chain_dict[player] += df_shot_sequence["shot_statsbomb_xg"].sum()

total_xg = df_all_shots["shot_statsbomb_xg"].sum()

#%%
# CALCULATE PASSING NETWORK

# load match data
min_num_passes = 4
schwiz_vs_spain = 3795108
slovakia_vs_spain = 3788775
sweden_vs_spain = 3788750
croatia_vs_spain = 3794686
italy_vs_spain = 3795220
parser = Sbopen()
df, related, freeze, tactics = parser.event(italy_vs_spain)

# check for index of first sub
sub = df.loc[df["type_name"] == "Substitution"].loc[df["team_name"] == team].iloc[0]["index"]
sub_time = df.loc[df["type_name"] == "Substitution"].loc[df["team_name"] == team].iloc[0]["minute"]
# make df with successfull passes by England until the first substitution
mask_team = ((df.type_name == 'Pass') &
                (df.team_name == team) &
                (df.index < sub) & (df.outcome_name.isnull()) &
                (df.sub_type_name != "Throw-in"))
# taking necessary columns
df_pass = df.loc[mask_team, ['x', 'y', 'end_x', 'end_y', "player_name", "pass_recipient_name"]]
# adjusting that only the surname of a player is presented.
# df_pass["player_name"] = df_pass["player_name"].apply(lambda x: str(x).split()[-1])
# df_pass["pass_recipient_name"] = df_pass["pass_recipient_name"].apply(lambda x: str(x).split()[-1])

# calculate average location of passes and pass receptions
scatter_df = pd.DataFrame()
for i, name in enumerate(df_pass["player_name"].unique()):
    passx = df_pass.loc[df_pass["player_name"] == name]["x"].to_numpy()
    recx = df_pass.loc[df_pass["pass_recipient_name"] == name]["end_x"].to_numpy()
    passy = df_pass.loc[df_pass["player_name"] == name]["y"].to_numpy()
    recy = df_pass.loc[df_pass["pass_recipient_name"] == name]["end_y"].to_numpy()
    scatter_df.at[i, "player_name"] = name
    #make sure that x and y location for each circle representing the player is the average of passes and receptions
    scatter_df.at[i, "x"] = np.mean(np.concatenate([passx, recx]))
    scatter_df.at[i, "y"] = np.mean(np.concatenate([passy, recy]))
    #calculate number of passes
    scatter_df.at[i, "no"] = df_pass.loc[df_pass["player_name"] == name].count().iloc[0]

#adjust the size of a circle so that the player who made more passes
scatter_df['marker_size'] = (scatter_df['no'] / scatter_df['no'].max() * 1500)

#counting passes between players
lines_df = df_pass.groupby(['player_name', 'pass_recipient_name']).x.count().reset_index()
lines_df.rename({'x':'pass_count'}, axis='columns', inplace=True)
#setting a threshold. You can try to investigate how it changes when you change it.
lines_df = lines_df[lines_df['pass_count'] > min_num_passes]

# calculate normalized pass count
column = "pass_count"
lines_df["pass_count_norm"] = (lines_df[column] - lines_df[column].min()) / (lines_df[column].max() - lines_df[column].min())    
lines_df.replace({'pass_count_norm': {0: 0.05}}, inplace=True)


## --------- CALCULATE CENTRALIZATION ------------ ##
# calculate number of successful passes by player
no_passes = df_pass.groupby(['player_name']).x.count().reset_index()
no_passes.rename({'x':'pass_count'}, axis='columns', inplace=True)
# find one who made most passes
max_no = no_passes["pass_count"].max()
# calculate the denominator - 10*the total sum of passes
denominator = 10*no_passes["pass_count"].sum()
# calculate the nominator
nominator = (max_no - no_passes["pass_count"]).sum()
# calculate the centralisation index
centralisation_index = nominator/denominator
print("Centralisation index is ", centralisation_index)


## --------- CALCULATE CENTRALIZATION FOR EACH PLAYER ------------- ##
#create list of nodes
nodes_final = list(lines_df["player_name"].unique())
edges_final = list(lines_df[["player_name", "pass_recipient_name", "pass_count"]].itertuples(index=False, name=None)) #create list of tuples (https://stackoverflow.com/questions/9758450/pandas-convert-dataframe-to-array-of-tuples)
#construct graph
g = nx.DiGraph()
g.add_nodes_from(nodes_final)
g.add_weighted_edges_from(edges_final)

#calculate betweeness centrality
between_centrality = nx.betweenness_centrality(g, weight='weight', normalized=True, endpoints=True)
centrality_df = pd.DataFrame(between_centrality.items(), columns=["player_name", "centrality"])

# merge with og dataframe
df_final = scatter_df.merge(centrality_df, on="player_name")


## --------- PLOT PLAYER PASSES --------- ##
mpl.rcParams['font.family'] = 'Alegreya Sans'

# draw pitch
vert_pitch = VerticalPitch(line_color="black", half=True, pitch_color="#efe9e6")
fig, ax = plt.subplots(figsize=(14,10))

fig.set_facecolor("#efe9e6")

## -------- PLOT PROG AND SHOT ASSISTS -------- ##
vert_pitch.draw(ax=ax)
# (remove shot assists from prog passes)
df_prog_passes = df_prog_passes[~(df_prog_passes.id.isin(df_player_shot_assist.id))]
# plot progressive passes
vert_pitch.lines(df_prog_passes.x, df_prog_passes.y,
                 df_prog_passes.end_x, df_prog_passes.end_y,
                 color = "k", ax=ax, comet=True, alpha=0.2, lw=12)
vert_pitch.scatter(df_prog_passes.end_x, df_prog_passes.end_y, ax=ax,
                   s = 500, color = "grey", edgecolor="k", lw=2,
                   zorder=3)
# highlight shot assists
vert_pitch.lines(df_shot_assist.x, df_shot_assist.y,
                 df_shot_assist.end_x, df_shot_assist.end_y,
                 color = "r", ax=ax, comet=True, lw=12, zorder=4)
vert_pitch.scatter(df_shot_assist.end_x, df_shot_assist.end_y, ax=ax,
                   alpha = 1, color = "r", edgecolor="k", lw=2,
                   zorder=5, s= df_shot_assist["shot_statsbomb_xg_shot"]*5000)

# Add global title
fig.text(x=0.5, y=0.98, ha="center",
         s="Pedri - EURO 2020",
         fontweight="bold", fontsize=42)
# subtitle to pass map
fig_text(x=0.5, y=0.94, s="All <Progressive passes> and <Shot assists>",
         fontsize=26, ha="center",
         highlight_textprops=[{"color": "grey"},
                              {"color": "r"}])

## PLOT PASSING NETWORK ##
bg = "#181818"
text_color = "#CECECD"
arrow_shift = 2.5

pitch = Pitch(line_color='grey')
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)


#plot pass arrows
for i, row in lines_df.iterrows():
    player1 = row["player_name"]
    player2 = row['pass_recipient_name']
    #take the average location of players to plot a line between them
    player1_x = scatter_df.loc[scatter_df["player_name"] == player1]['x'].iloc[0]
    player1_y = scatter_df.loc[scatter_df["player_name"] == player1]['y'].iloc[0]
    player2_x = scatter_df.loc[scatter_df["player_name"] == player2]['x'].iloc[0]
    player2_y = scatter_df.loc[scatter_df["player_name"] == player2]['y'].iloc[0]
    num_passes = row["pass_count"]
    pass_count_norm = row["pass_count_norm"]
    
    if abs(player2_x - player1_x) > abs(player2_y - player1_y):

        if player1 > player2:
            draw_arrow_with_shrink(ax=ax["pitch"], x=player1_x, y=player1_y + arrow_shift,
                       end_x=player2_x, end_y=player2_y,
                       lw=3, line_color='k', 
                       alpha=pass_count_norm, dist_delta=4)

        elif player2 > player1:
            draw_arrow_with_shrink(ax=ax["pitch"], x=player1_x, y=player1_y - arrow_shift,
                       end_x=player2_x, end_y=player2_y,
                       lw=3, line_color='k', 
                       alpha=pass_count_norm, dist_delta=4)

    elif abs(player2_x - player1_x) <= abs(player2_y - player1_y):

        if player1 > player2:
             draw_arrow_with_shrink(ax=ax["pitch"], x=player1_x + arrow_shift, y=player1_y,
                       end_x=player2_x, end_y=player2_y,
                       lw=3, line_color='k', 
                       alpha=pass_count_norm, dist_delta=4)
        
        elif player2 > player1:
            draw_arrow_with_shrink(ax=ax["pitch"], x=player1_x - arrow_shift, y=player1_y,
                       end_x=player2_x, end_y=player2_y,
                       lw=3, line_color='k', 
                       alpha=pass_count_norm, dist_delta=4)

#plot the nodes
nodes = ax["pitch"].scatter(scatter_df.x, scatter_df.y,
                   s=scatter_df.marker_size, color='red', edgecolor='k', 
                   linewidth=2.5, alpha=1, zorder=3)
for i, row in scatter_df.iterrows():
    pitch.annotate(row.player_name, xy=(row.x, row.y),
                   c='black', va='center', ha='center',
                   weight = "bold", size=16, ax=ax["pitch"], zorder = 4)

# title etc.
# title
fig.text(x=0.5, y=1, ha="center",
         fontsize=42, fontweight="bold",
         s="Spain passing network")
fig.text(x=0.5,y=0.94, ha="center",
         fontsize=24, fontweight="regular",
         s="Spain vs. Italy | Semi-final | EURO 2020")

# corner text
fig.text(x=0.85, y=0, ha="left", fontsize=12,# fonweight="italic",
         s=f"*until first sub at {sub_time} min.\n*min. {min_num_passes} number of passes")