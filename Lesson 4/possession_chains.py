# Lesson 4: Possesion Chains
# https://soccermatics.readthedocs.io/en/latest/gallery/lesson4/plot_PossesionChain.html
#%%
import pandas as pd
import numpy as np
import json
import statsmodels.api as sm
import statsmodels.formula.api as smf
from mplsoccer import Pitch
import matplotlib.pyplot as plt

from utils import isolateChains, calulatexG, prepareChains

# Load data
with open("../data/wyscout/events/events_England.json") as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Prepare data
# get next event
next_event = df.shift(-1, fill_value=0)
df["nextEvent"] = next_event["subEventName"]
df["kickedOut"] = df.apply(lambda x: 1 if x.nextEvent == "Ball out of the field" else 0, axis = 1)
# remove interruptions
interruption = df.loc[df["eventName"] == "Interruption"]
# filter out non-accurate duels - in wyscout they are 2 way - attacking and defending
lost_duels = df.loc[df["eventName"] == "Duel"]
lost_duels = lost_duels.loc[lost_duels.apply(lambda x:{'id': 1802} in x.tags, axis = 1)]
df = df.drop(lost_duels.index)
# filter ball out of the field
out_of_field = df.loc[df["subEventName"] == "Ball out of the field"]
df = df.drop(out_of_field.index)
# save attempts can be dropped
goalies = df.loc[df["subEventName"].isin(["Goalkeeper leaving line", "Save attempt", "Reflexes"])]
df = df.drop(goalies.index)

# Isolate Possession Chains
# poss chain ends with foul, ball out of play and if opp. team has two touches on the ball
df = isolateChains(df)
# df.loc[df["possesion_chain"] == 4][["eventName", "possesion_chain"]]

# Calculate xG
df = calulatexG(df)
# df.loc[df["possesion_chain"].isin([3,4])][["eventName", "possesion_chain", "xG"]]
#%%

# Find chains that ended with shot
# assign 1 to chains ending in a shot, and assign xG value of shot to all actions in the chain
df = prepareChains(df)
df.loc[df["possesion_chain"].isin([3,4])][["eventName", "possesion_chain", "xG"]]

# Prepare data for model
# -> remove events with no end coordinates
# -> create separate coordinate columns
# -> calculate orthogonal distance to the halfway line
# -> move shot end location to 105, 34 (in the goal rather than the corner...)

# remove events with no end coordinates
df = df.loc[df.apply(lambda x: len(x.positions) == 2, axis = 1)]
# separate coordinates
df["x0"] = df.positions.apply(lambda cell: (cell[0]['x']) * 105/100)
df["x1"] = df.positions.apply(lambda cell: (cell[1]['x']) * 105/100)
# calculare orthogonal line
df["c0"] = df.positions.apply(lambda cell: abs(50 - cell[0]['y']) * 68/100)
df["c1"] = df.positions.apply(lambda cell: abs(50 - cell[1]['y']) * 68/100)
# assign (105, 34) to end of the shot
df.loc[df["eventName"] == "Shot", "x1"] = 105
df.loc[df["eventName"] == "Shot", "c1"] = 0

#for plotting
df["y0"] = df.positions.apply(lambda cell: (100 - cell[0]['y']) * 68/100)
df["y1"] = df.positions.apply(lambda cell: (100 - cell[1]['y']) * 68/100)
df.loc[df["eventName"] == "Shot", "y1"] = 34

#%%
# Plot possession chain that ended in a shot
#plot possesion chain that ended with shot
chain = df.loc[df["possesion_chain"] == 4]
#get passes
passes = chain.loc[chain["eventName"].isin(["Pass"])]
#get events different than pass
not_pass = chain.loc[chain["eventName"] != "Pass"].iloc[:-1]
#shot is the last event of the chain (or should be)
shot = chain.iloc[-1]
#plot
pitch = Pitch(line_color='black',pitch_type='custom', pitch_length=105, pitch_width=68, line_zorder = 2)
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
#passes
pitch.arrows(passes.x0, passes.y0,
            passes.x1, passes.y1, color = "blue", ax=ax['pitch'], zorder =  3)
#shot
pitch.arrows(shot.x0, shot.y0,
             shot.x1, shot.y1, color = "red", ax=ax['pitch'], zorder =  3)
#other passes like arrows
pitch.lines(not_pass.x0, not_pass.y0, not_pass.x1, not_pass.y1, color = "grey", lw = 1.5, ls = 'dotted', ax=ax['pitch'])
ax['title'].text(0.5, 0.5, 'Passes leading to a shot', ha='center', va='center', fontsize=30)

#%%
# Plot possession that did not end in a shot
#plot possesion chain that ended with shot
chain = df.loc[df["possesion_chain"] == 0]
passes = chain.loc[chain["eventName"].isin(["Pass", "Free Kick"])].iloc[:-1]
not_pass = chain.loc[(chain["eventName"] != "Pass") & (chain["eventName"] != "Free Kick")].iloc[:-1]
bad_pass = chain.iloc[-1]
#we could have used better
pitch = Pitch(line_color='black',pitch_type='custom', pitch_length=105, pitch_width=68, line_zorder = 2)
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
pitch.arrows(passes.x0, passes.y0,
            passes.x1, passes.y1, color = "blue", ax=ax['pitch'], zorder =  3)
pitch.arrows(bad_pass.x0, bad_pass.y0,
            bad_pass.x1, bad_pass.y1, color = "purple", ax=ax['pitch'], zorder =  3)
pitch.scatter(bad_pass.x1, bad_pass.y1, marker = 'x', color = "red", ax=ax['pitch'], zorder =  3, s= 200)
pitch.lines(not_pass.x0, not_pass.y0, not_pass.x1, not_pass.y1, color = "grey", lw = 1.5, ls = 'dotted', ax=ax['pitch'])
ax['title'].text(0.5, 0.5, 'Passes not ending in a shot', ha='center', va='center', fontsize=30)
plt.show()