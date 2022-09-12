#%%
# Lesson 2 - Linear Regression
# https://soccermatics.readthedocs.io/en/latest/gallery/lesson2/plot_LinearRegression.html

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Load data
laliga_df = pd.read_csv("data/player_stats.csv")

# Only take first 20 observations, and only use minutes and age for fitting
num_obs = 20
minutes_model = pd.DataFrame()
minutes_model = minutes_model.assign(minutes=laliga_df["Min"][0:num_obs])
minutes_model = minutes_model.assign(age=laliga_df["Age"][0:num_obs])

# Make an age squared column so we can fit a polynomial model.
minutes_model = minutes_model.assign(age_squared=np.power(laliga_df["Age"][0:num_obs], 2))

# Fitting the linear model
model_fit = smf.ols(formula='minutes  ~ age   ', data=minutes_model).fit()
b = model_fit.params

# Plotting the data
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(minutes_model['age'], minutes_model['minutes'], linestyle='none', marker= '.', markersize= 10, color='blue')
ax.set_ylabel('Minutes played')
ax.set_xlabel('Age')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim((15,40))
plt.ylim((0,3000))

# Now create the fitting line through the data
x = np.arange(40,step=1)
y = b[0] + b[1]*x
# y = np.mean(minutes_model['minutes'])*np.ones(40)
ax.plot(x, y, color='black')

#Show distances to line for each point
for i, a in enumerate(minutes_model['age']):
    ax.plot([a,a],
            [minutes_model['minutes'][i], b[0] + b[1]*a],
            color='red', zorder=-1)
fig.suptitle("Linear Model", fontsize=18, fontweight="bold")
#%%
# Fitting the non-linear model
model_fit = smf.ols(formula='minutes  ~ age + age_squared  ', data=minutes_model).fit()
b = model_fit.params

# Compare the fit
fig, ax=plt.subplots(figsize=(8,6))
ax.plot(minutes_model['age'], minutes_model['minutes'],
        linestyle='none', marker= '.', markersize= 10, color='blue')
ax.set_ylabel('Minutes played')
ax.set_xlabel('Age')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim((15,40))
plt.ylim((0,3000))

x = np.arange(40,step=1)
y = b[0] + b[1]*x + b[2]*x*x

ax.plot(x, y, color='black')

for i, a in enumerate(minutes_model['age']):
    ax.plot([a,a],
            [minutes_model['minutes'][i], b[0] + b[1]*a + b[2]*a*a],
            color='red', zorder=-1)
fig.suptitle("Non-linear Model", fontsize=18, fontweight="bold")

#%%
## CHALLENGE ##
# 1. Refit the model with all data points
# add all data
minutes_model_all = pd.DataFrame()
minutes_model_all = minutes_model_all.assign(minutes=laliga_df["Min"])
minutes_model_all = minutes_model_all.assign(age=laliga_df["Age"])
# add quadratic term
minutes_model_all = minutes_model_all.assign(age_squared = np.power(laliga_df["Age"], 2))
# fit model
model_fit = smf.ols(formula='minutes  ~ age + age_squared  ', data=minutes_model_all).fit()
b = model_fit.params

# 2. Try adding a cubic term
minutes_model_all = minutes_model_all.assign(age_cubed = np.power(laliga_df["Age"], 3))
model_fit_cubic = smf.ols(formula='minutes  ~ age + age_cubed  ', data=minutes_model_all).fit()
b = model_fit_cubic.params

# 3. Think about how well the model works. What are the limitations?