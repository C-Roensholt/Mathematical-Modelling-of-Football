#%%
import math
import numpy as np
import networkx as nx
import pandas as pd
from mplsoccer import Pitch, Sbopen
import matplotlib.pyplot as plt

# Load data
parser = Sbopen()
df, related, freeze, tactics = parser.event(69301)

# check for index of first sub
sub = df.loc[df["type_name"] == "Substitution"].loc[df["team_name"] == "England Women's"].iloc[0]["index"]
# make df with successfull passes by England until the first substitution
mask_england = ((df.type_name == 'Pass') &
                (df.team_name == "England Women's") &
                (df.index < sub) & (df.outcome_name.isnull()) &
                (df.sub_type_name != "Throw-in"))
# taking necessary columns
df_pass = df.loc[mask_england, ['x', 'y', 'end_x', 'end_y', "player_name", "pass_recipient_name"]]
# adjusting that only the surname of a player is presented.
df_pass["player_name"] = df_pass["player_name"].apply(lambda x: str(x).split()[-1])
df_pass["pass_recipient_name"] = df_pass["pass_recipient_name"].apply(lambda x: str(x).split()[-1])

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
lines_df = lines_df[lines_df['pass_count']>2]

# calculate normalized pass count
column = "pass_count"
lines_df["pass_count_norm"] = (lines_df[column] - lines_df[column].min()) / (lines_df[column].max() - lines_df[column].min())    
lines_df.replace({'pass_count_norm': {0: 0.05}}, inplace=True)

#Drawing pitch
pitch = Pitch(line_color='grey')
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
pitch.scatter(scatter_df.x, scatter_df.y, s=scatter_df.marker_size,
              color='red', edgecolor='black', linewidth=1, alpha=1, ax=ax["pitch"], zorder = 3)
for i, row in scatter_df.iterrows():
    pitch.annotate(row.player_name, xy=(row.x, row.y), c='black', va='center', ha='center', weight = "bold", size=16, ax=ax["pitch"], zorder = 4)

for i, row in lines_df.iterrows():
        player1 = row["player_name"]
        player2 = row['pass_recipient_name']
        #take the average location of players to plot a line between them
        player1_x = scatter_df.loc[scatter_df["player_name"] == player1]['x'].iloc[0]
        player1_y = scatter_df.loc[scatter_df["player_name"] == player1]['y'].iloc[0]
        player2_x = scatter_df.loc[scatter_df["player_name"] == player2]['x'].iloc[0]
        player2_y = scatter_df.loc[scatter_df["player_name"] == player2]['y'].iloc[0]
        num_passes = row["pass_count"]
        #adjust the line width so that the more passes, the wider the line
        line_width = (num_passes / lines_df['pass_count'].max() * 15)
        #plot lines on the pitch
        pitch.lines(player1_x, player1_y, player2_x, player2_y,
                        alpha=1, lw=line_width, zorder=2, color="red", ax = ax["pitch"])

fig.suptitle("England Passing Network against Sweden", fontsize = 30)

##%%
## ------------ CUSTOM PASSING NETWORK (ARROWS WITH SHIFT) --------------- ##
def draw_arrow_with_shrink(ax, x, y, end_x, end_y, lw, line_color, alpha, dist_delta=2.5):
    dist = math.hypot(end_x - x, end_y - y)
    angle = math.atan2(end_y - y, end_x - x)
    upd_end_x = x + (dist - dist_delta) * math.cos(angle)
    upd_end_y = y + (dist - dist_delta) * math.sin(angle)
    upd_x = end_x - (dist - dist_delta * 1.2) * math.cos(angle)
    upd_y = end_y - (dist - dist_delta * 1.2) * math.sin(angle)
    ax.annotate('', xy=(upd_end_x, upd_end_y), xytext=(upd_x, upd_y), zorder=1,
                arrowprops=dict(linewidth=lw, color=line_color, alpha=alpha,
                                headwidth=15, headlength=15))
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

#lines.set_offsets(2.1)

#plot the nodes
nodes = ax["pitch"].scatter(scatter_df.x, scatter_df.y,
                   s=scatter_df.marker_size, color='red', edgecolor='k', 
                   linewidth=2.5, alpha=1, zorder=3)
for i, row in scatter_df.iterrows():
    pitch.annotate(row.player_name, xy=(row.x, row.y),
                   c='black', va='center', ha='center',
                   weight = "bold", size=16, ax=ax["pitch"], zorder = 4)

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


## --------- CHALLENGE (PASSING NETWORK WITH FORWARD PASSES) ---------- ##
# filter for forward passes
mask_foward = (df.end_x > df.x)
mask_long = (df.end_x - df.x > 20)
df_pass_forward = df_pass.loc[mask_foward, :]
df_pass_long = df_pass.loc[mask_long, :]

# calculate average location of passes and pass receptions
scatter_df = pd.DataFrame()
for i, name in enumerate(df_pass_forward["player_name"].unique()):
    passx = df_pass_forward.loc[df_pass_forward["player_name"] == name]["x"].to_numpy()
    recx = df_pass_forward.loc[df_pass_forward["pass_recipient_name"] == name]["end_x"].to_numpy()
    passy = df_pass_long.loc[df_pass_forward["player_name"] == name]["y"].to_numpy()
    recy = df_pass_forward.loc[df_pass_forward["pass_recipient_name"] == name]["end_y"].to_numpy()
    scatter_df.at[i, "player_name"] = name
    #make sure that x and y location for each circle representing the player is the average of passes and receptions
    scatter_df.at[i, "x"] = np.mean(np.concatenate([passx, recx]))
    scatter_df.at[i, "y"] = np.mean(np.concatenate([passy, recy]))
    #calculate number of passes
    scatter_df.at[i, "no"] = df_pass_forward.loc[df_pass_forward["player_name"] == name].count().iloc[0]

#adjust the size of a circle so that the player who made more passes
scatter_df['marker_size'] = (scatter_df['no'] / scatter_df['no'].max() * 1500)

#counting passes between players
lines_df = df_pass_forward.groupby(['player_name', 'pass_recipient_name']).x.count().reset_index()
lines_df.rename({'x':'pass_count'}, axis='columns', inplace=True)
#setting a threshold. You can try to investigate how it changes when you change it.
# lines_df = lines_df[lines_df['pass_count']>2]

# calculate normalized pass count
column = "pass_count"
lines_df["pass_count_norm"] = (lines_df[column] - lines_df[column].min()) / (lines_df[column].max() - lines_df[column].min())    
lines_df.replace({'pass_count_norm': {0: 0.05}}, inplace=True)

#Drawing pitch
pitch = Pitch(line_color='grey')
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
pitch.scatter(scatter_df.x, scatter_df.y, s=scatter_df.marker_size,
              color='red', edgecolor='black', linewidth=1, alpha=1, ax=ax["pitch"], zorder = 3)
for i, row in scatter_df.iterrows():
    pitch.annotate(row.player_name, xy=(row.x, row.y), c='black', va='center', ha='center', weight = "bold", size=16, ax=ax["pitch"], zorder = 4)

for i, row in lines_df.iterrows():
        player1 = row["player_name"]
        player2 = row['pass_recipient_name']
        #take the average location of players to plot a line between them
        player1_x = scatter_df.loc[scatter_df["player_name"] == player1]['x'].iloc[0]
        player1_y = scatter_df.loc[scatter_df["player_name"] == player1]['y'].iloc[0]
        player2_x = scatter_df.loc[scatter_df["player_name"] == player2]['x'].iloc[0]
        player2_y = scatter_df.loc[scatter_df["player_name"] == player2]['y'].iloc[0]
        num_passes = row["pass_count"]
        #adjust the line width so that the more passes, the wider the line
        line_width = (num_passes / lines_df['pass_count'].max() * 15)
        #plot lines on the pitch
        pitch.lines(player1_x, player1_y, player2_x, player2_y,
                        alpha=1, lw=line_width, zorder=2, color="red", ax = ax["pitch"])

fig.suptitle("England Passing Network against Sweden", fontsize = 30)