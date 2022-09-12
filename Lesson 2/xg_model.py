#%%
# https://soccermatics.readthedocs.io/en/latest/gallery/lesson2/plot_xGModelFit.html#sphx-glr-gallery-lesson2-plot-xgmodelfit-py
# importing necessary libraries
import pandas as pd
import numpy as np
import json
# plotting
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
# statistical fitting of models
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
with open("../data/wyscout/events/events_England.json") as f:
    data = json.load(f)
train = pd.DataFrame(data)
#%%
# Prep for xG modelling
#get shots
shots = train.loc[train['subEventName'] == 'Shot']
#get shot coordinates into separate columns
shots["x"] = shots.positions.apply(lambda cell: (100 - cell[0]['x']) * 105/100)
shots["y"] = shots.positions.apply(lambda cell: cell[0]['y'] * 68/100)
# c is an auxillary variable to help us calculate distance and angle
# the distance from a point to the vertical line through the middle of the pitch
shots["c"] = shots.positions.apply(lambda cell: abs(cell[0]['y'] - 50) * 68/100)
#calculate distance and angle to goal
shots["distance"] = np.sqrt(shots["x"]**2 + shots["c"]**2)
shots["angle"] = (np.where(np.arctan(7.32 * shots["x"] / (shots["x"]**2 + shots["c"]**2 - (7.32/2)**2)) > 0,
                           np.arctan(7.32 * shots["x"] / (shots["x"]**2 + shots["c"]**2 - (7.32/2)**2)),
                           np.arctan(7.32 * shots["x"] / (shots["x"]**2 + shots["c"]**2 - (7.32/2)**2)) + np.pi))
#if you ever encounter problems (like you have seen that model treats 0 as 1 and 1 as 0) while modelling - change the dependant variable to object
shots["goal"] = shots.tags.apply(lambda x: 1 if {'id':101} in x else 0).astype(object)

#%%
# PLOT SHOT LOCATION
#plot pitch
pitch = VerticalPitch(line_color='black', half = True, pitch_type='custom', pitch_length=105, pitch_width=68, line_zorder = 2)
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
#subtracting x from 105 but not y from 68 because of inverted Wyscout axis
#calculate number of shots in each bin
bin_statistic_shots = pitch.bin_statistic(105 - shots.x, shots.y, bins=50)
#make heatmap
pcm = pitch.heatmap(bin_statistic_shots, ax=ax["pitch"], cmap='Reds', edgecolor='white', linewidth = 0.01)
#make legend
ax_cbar = fig.add_axes((0.95, 0.05, 0.04, 0.8))
cbar = plt.colorbar(pcm, cax=ax_cbar)
fig.suptitle('Shot map - 2017/2018 Premier League Season' , fontsize = 30)
plt.show()

#%%
# PLOT GOAL LOCATION
#take only goals
goals = shots.loc[shots["goal"] == 1]
#plot pitch
pitch = VerticalPitch(line_color='black', half = True, pitch_type='custom', pitch_length=105, pitch_width=68, line_zorder = 2)
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
#calculate number of goals in each bin
bin_statistic_goals = pitch.bin_statistic(105 - goals.x, goals.y, bins=50)
#plot heatmap
pcm = pitch.heatmap(bin_statistic_goals, ax=ax["pitch"], cmap='Reds', edgecolor='white')
#make legend
ax_cbar = fig.add_axes((0.95, 0.05, 0.04, 0.8))
cbar = plt.colorbar(pcm, cax=ax_cbar)
fig.suptitle('Goal map - 2017/2018 Premier League Season' , fontsize = 30)
plt.show()

#%%
# PROPBABILITY OF SCORING FROM GIVEN LOCATION
#plot pitch
pitch = VerticalPitch(line_color='black', half = True, pitch_type='custom', pitch_length=105, pitch_width=68, line_zorder = 2)
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
bin_statistic = pitch.bin_statistic(105 - shots.x, shots.y, bins = 50)
#normalize number of goals by number of shots
bin_statistic["statistic"] = bin_statistic_goals["statistic"] / bin_statistic["statistic"]
#plot heatmap
pcm = pitch.heatmap(bin_statistic, ax=ax["pitch"], cmap='Reds', edgecolor='white', vmin = 0, vmax = 0.6)
#make legend
ax_cbar = fig.add_axes((0.95, 0.05, 0.04, 0.8))
cbar = plt.colorbar(pcm, cax=ax_cbar)
fig.suptitle('Probability of scoring' , fontsize = 30)
plt.show()

#%%
# PLOT LOGISTIC CURVE
b = [3, -3]
x = np.arange(5, step=0.1)
y = 1 / (1 + np.exp(b[0] + b[1]*x))
fig,ax = plt.subplots()
plt.ylim((-0.05, 1.05))
plt.xlim((0,5))
ax.set_ylabel("y")
ax.set_xlabel("x")
ax.plot(x, y, linestyle='solid', color='black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()

#%%
# RELATIONSHIP BETWEEN GOALS AND ANGLE
#first 200 shots
shots_200 = shots.iloc[:200]
#plot first 200 shots goal angle
fig, ax = plt.subplots()
ax.plot(shots_200['angle']*180 / np.pi, shots_200['goal'],
        linestyle='none', marker= '.', markersize= 12, color='black')
#make legend
ax.set_ylabel('Goal scored')
ax.set_xlabel("Shot angle (degrees)")
plt.ylim((-0.05,1.05))
ax.set_yticks([0,1])
ax.set_yticklabels(['No','Yes'])
plt.show()

#%%
# RELATIONSHIP BETWEEN PROP OF SCORING AND ANGLE
#number of shots from angle
shotcount_dist = np.histogram(shots['angle']*180/np.pi, bins=40, range=[0, 150])
#number of goals from angle
goalcount_dist = np.histogram(goals['angle']*180/np.pi, bins=40, range=[0, 150])
np.seterr(divide='ignore', invalid='ignore')
#probability of scoring goal
prob_goal = np.divide(goalcount_dist[0], shotcount_dist[0])
angle = shotcount_dist[1]
midangle = (angle[:-1] + angle[1:])/2

#make plot
fig,ax = plt.subplots()
ax.plot(midangle, prob_goal, linestyle='none', marker= '.', markersize= 12, color='black')
ax.set_ylabel('Probability chance scored')
ax.set_xlabel("Shot angle (degrees)")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()

#%%
# FIT LOGISTIC REGRESSION
fig, ax = plt.subplots()
b = [3, -3]
x = np.arange(150, step=0.1)
y = 1 / (1+np.exp(b[0] + b[1]*x*np.pi / 180))
#plot line
ax.plot(midangle, prob_goal, linestyle='none', marker= '.', markersize= 12, color='black')
#plot logistic function
ax.plot(x, y, linestyle='solid', color='black')
plt.show()

#%%
# CALCULATE LOG LIKELIHOOD
#calculate xG
xG = 1/(1+np.exp(b[0]+b[1]*shots['angle']))
shots = shots.assign(xG = xG)
shots_40 = shots.iloc[:40]
fig, ax = plt.subplots()
#plot data
ax.plot(shots_40['angle']*180/np.pi, shots_40['goal'], linestyle='none', marker= '.', markersize= 12, color='black', zorder = 3)
#plot curves
ax.plot(x, y, linestyle=':', color='black', zorder = 2)
ax.plot(x, 1-y, linestyle='solid', color='black', zorder = 2)
#calculate loglikelihood
loglikelihood = 0
for item,shot in shots_40.iterrows():
    ang = shot['angle'] * 180/np.pi
    if shot['goal'] == 1:
        loglikelihood = loglikelihood + np.log(shot['xG'])
        ax.plot([ang,ang],[shot['goal'],1-shot['xG']], color='red', zorder = 1)
    else:
        loglikelihood = loglikelihood + np.log(1 - shot['xG'])
        ax.plot([ang,ang], [shot['goal'], 1-shot['xG']], color='blue', zorder = 1)
#make legend
ax.set_ylabel('Goal scored')
ax.set_xlabel("Shot angle (degrees)")
plt.ylim((-0.05,1.05))
plt.xlim((0,180))
plt.text(120,0.5,'Log-likelihood:')
plt.text(120,0.4,str(loglikelihood)[:6])
ax.set_yticks([0,1])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()

#%%
# FIT LOGISTIC REGRESSION
# The best parametrs maximize log-likelihood
#create model
test_model = smf.glm(formula="goal ~ angle" , data=shots,
                           family=sm.families.Binomial()).fit()
print(test_model.summary())
#get params
b=test_model.params
#calculate xG
xGprob = 1/(1+np.exp(b[0]+b[1]*midangle*np.pi/180))
fig, ax = plt.subplots()
#plot data
ax.plot(midangle, prob_goal, linestyle='none', marker= '.', markersize= 12, color='black')
#plot line
ax.plot(midangle, xGprob, linestyle='solid', color='black')
#make legend
ax.set_ylabel('Probability chance scored')
ax.set_xlabel("Shot angle (degrees)")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()

#%%
# RELATIONSHIP BETWEEN PROP OF GOAL AND DIST TO GOAL
#number of shots
shotcount_dist = np.histogram(shots['distance'], bins=40, range=[0, 70])
#number of goals
goalcount_dist = np.histogram(goals['distance'], bins=40, range=[0, 70])
#empirical probability of scoring
prob_goal = np.divide(goalcount_dist[0],shotcount_dist[0])
distance = shotcount_dist[1]
middistance= (distance[:-1] + distance[1:]) / 2
#making a plot
fig, ax = plt.subplots()
#plotting data
ax.plot(middistance, prob_goal, linestyle='none', marker= '.', color='black')
#making legend
ax.set_ylabel("Probability chance scored")
ax.set_xlabel("Distance from goal (metres)")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# fit logistic model
#make single variable model of distance
test_model = smf.glm(formula="goal ~ distance" , data=shots,
                     family=sm.families.Binomial()).fit()
#print summary
print(test_model.summary())
b = test_model.params
#calculate xG
xGprob = 1 / (1+np.exp(b[0]+b[1]*middistance))
#plot line
ax.plot(middistance, xGprob, linestyle='solid', color='black')
plt.show()

#%%




# ADD SQUADRED DISTANCE TO OUR MODEL
#creating extra variables
shots["x2"] = shots['x']**2
shots["c2"] = shots['c']**2
shots["ax"]  = shots['angle']*shots['x']

# list the model variables you want here
model_variables = ['angle', 'distance', 'x', 'c', "x2", "c2", "ax"]
model=''
for v in model_variables[:-1]:
    model = model  + v + ' + '
model = model + model_variables[-1]

#fit the model
test_model = smf.glm(formula="goal ~ " + model, data=shots,
                     family=sm.families.Binomial()).fit()
#print summary
print(test_model.summary())
b = test_model.params

#return xG value for more general model
def calculate_xG(sh):
   bsum=b[0]
   for i,v in enumerate(model_variables):
       bsum=bsum+b[i+1]*sh[v]
   xG = 1/(1+np.exp(bsum))
   return xG

#add an xG to my dataframe
xG = shots.apply(calculate_xG, axis=1)
shots = shots.assign(xG=xG)

#Create a 2D map of xG
pgoal_2d = np.zeros((68,68))
for x in range(68):
    for y in range(68):
        sh=dict()
        a = np.arctan(7.32 *x /(x**2 + abs(y-68/2)**2 - (7.32/2)**2))
        if a<0:
            a = np.pi + a
        sh['angle'] = a
        sh['distance'] = np.sqrt(x**2 + abs(y-68/2)**2)
        sh['d2'] = x**2 + abs(y-68/2)**2
        sh['x'] = x
        sh['ax'] = x*a
        sh['x2'] = x**2
        sh['c'] = abs(y-68/2)
        sh['c2'] = (y-68/2)**2

        pgoal_2d[x,y] =  calculate_xG(sh)

#plot pitch
pitch = VerticalPitch(line_color='black', half = True,
                      pitch_type='custom', pitch_length=105, pitch_width=68, line_zorder = 2)
fig, ax = pitch.draw()
#plot probability
pos = ax.imshow(pgoal_2d, extent=[-1,68,68,-1], aspect='auto',
                cmap=plt.cm.Reds,vmin=0, vmax=0.3, zorder = 1)
fig.colorbar(pos, ax=ax)
#make legend
ax.set_title('Probability of goal')
plt.xlim((0,68))
plt.ylim((0,60))
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

#%%
# TEST MODEL FIT
# Mcfaddens Rsquared for Logistic regression
null_model = smf.glm(formula="goal ~ 1 ", data=shots,
                     family=sm.families.Binomial()).fit()
print("Mcfaddens Rsquared", 1 - test_model.llf / null_model.llf)

# ROC curve
numobs = 100
TP = np.zeros(numobs)
FP = np.zeros(numobs)
TN = np.zeros(numobs)
FN = np.zeros(numobs)

for i, threshold in enumerate(np.arange(0, 1, 1 / numobs)):
    for j, shot in shots.iterrows():
        if (shot['goal'] == 1):
            if (shot['xG'] > threshold):
                TP[i] = TP[i] + 1
            else:
                FN[i] = FN[i] + 1
        if (shot['goal'] == 0):
            if (shot['xG'] > threshold):
                FP[i] = FP[i] + 1
            else:
                TN[i] = TN[i] + 1

fig, ax = plt.subplots()
ax.plot(FP / (FP + TN), TP / (TP + FN), color='black')
ax.plot([0, 1], [0, 1], linestyle='dotted', color='black')
ax.set_ylabel("Predicted to score and did TP/(TP+FN))")
ax.set_xlabel("Predicted to score but didn't FP/(FP+TN)")
plt.ylim((0.00, 1.00))
plt.xlim((0.00, 1.00))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

## CHALLENGE ##
# 1. Create different models for headers and non-headers (as suggested in Measuring the Effectiveness of Playing Strategies at Soccer, Pollard (1997))!

# 2. Assign to penalties xG = 0.8!

# 3. Find out which player had the highest xG in 2017/18 Premier League season!