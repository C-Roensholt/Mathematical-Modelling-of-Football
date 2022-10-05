#%%
# Lesson 5 - Simulating Results
# https://soccermatics.readthedocs.io/en/latest/gallery/lesson5/plot_SimulateMatches.html

# importing libraries
import seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson,skellam

# import data
epl = pd.read_csv("https://www.football-data.co.uk/mmz4281/2122/E0.csv")
ep = epl[["HomeTeam","AwayTeam","FTHG","FTAG"]]
epl = epl.rename(columns={"FTHG": "HomeGoals", "FTAG": "AwayGoals"})
# subset data
epl = epl[:-10]

# fit model
epl_home = (epl[["HomeTeam","AwayTeam","HomeGoals"]].assign(home=1)
            .rename(columns={"HomeTeam":"team", "AwayTeam":"opponent","HomeGoals":"goals"}))
epl_away = (epl[["AwayTeam","HomeTeam","AwayGoals"]].assign(home=0)
            .rename(columns={"AwayTeam":"team", "HomeTeam":"opponent","AwayGoals":"goals"}))
goal_model_data = pd.concat([epl_home, epl_away])

poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data,
                        family=sm.families.Poisson()).fit()
# poisson_model.summary()

# simulate a game between City and Arsenal
# et teams here
home_team = "Man City"
away_team = "Arsenal"

# run prediction
df_home = pd.DataFrame(data={"team": home_team,
                             "opponent": away_team, "home": 1}, index=[1])
df_away = pd.DataFrame(data={"team": away_team,
                             "opponent": home_team, "home": 1}, index=[1])
home_score_rate = poisson_model.predict(df_home)
away_score_rate = poisson_model.predict(df_away)
print(home_team + " against " + away_team + " expected goals: " + str(round(home_score_rate.values[0], 2)))
print(away_team + " against " + home_team + " expected goals: " + str(round(away_score_rate.values[0], 2)))

#Lets just get a result
home_goals = np.random.poisson(home_score_rate)
away_goals = np.random.poisson(away_score_rate)
print(home_team + ": " + str(home_goals[0]))
print(away_team + ": "  + str(away_goals[0]))


# Create 2D histogram of probabilities of goals
# caluclate goals for the match.
def simulate_match(foot_model, homeTeam, awayTeam, max_goals=10):
    home_goals_avg = foot_model.predict(pd.DataFrame(data={"team": homeTeam,
                                                           "opponent": awayTeam, "home": 1},
                                                     index=[1])).values[0]
    away_goals_avg = foot_model.predict(pd.DataFrame(data={"team": awayTeam,
                                                           "opponent": homeTeam, "home": 0},
                                                     index=[1])).values[0]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals + 1)] for team_avg in
                 [home_goals_avg, away_goals_avg]]
    return (np.outer(np.array(team_pred[0]), np.array(team_pred[1])))

# fill matrix
max_goals = 5
score_matrix = simulate_match(poisson_model, home_team, away_team,max_goals)
# plotting
fig, ax = plt.subplots(figsize=(6, 4))
pos = ax.imshow(score_matrix,
                extent=[-0.5, max_goals+0.5, -0.5, max_goals+0.5],
                aspect="auto", cmap=plt.cm.Reds)
fig.colorbar(pos, ax=ax)
# formatting
plt.xlim((-0.5,5.5))
plt.ylim((-0.5,5.5))
plt.tight_layout()
# labels and title
ax.set_title("Probability of outcome")
ax.set_xlabel("Goals scored by " + away_team)
ax.set_ylabel("Goals scored by " + home_team)

# home, draw, away probabilities
homewin = np.sum(np.tril(score_matrix, -1))
draw = np.sum(np.diag(score_matrix))
awaywin = np.sum(np.triu(score_matrix, 1))