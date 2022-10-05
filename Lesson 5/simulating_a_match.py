# Lesson 5
# https://soccermatics.readthedocs.io/en/latest/gallery/lesson5/plot_simulatematch.html
#%%
import time
import numpy as np
import pylab as plt
import numpy.random as rnd

# length of match
match_minutes = 90
# average goals per match
goals_per_match = 2.79
# probability of a goal per minute
prob_per_minute = np.array(goals_per_match/match_minutes)
print(f"The probability of a goal per minute is {prob_per_minute}. \n")

## SIMULATING A SINGLE GAME ##
# count of the number of goals
goals = 0
for minute in range(match_minutes):
  # generate random number between 0 and 1
  r = rnd.rand(1, 1)
  # prints an X when there is a goal and a zero otherwise
  if (r < prob_per_minute):
    # goal - if the random number is less than the goal probability
    print("X", end = " ")
    goals = goals + 1
    time.sleep(1)
  else:
    print("o", end = " ")
    time.sleep(0.1)
print("\n")
print(f"Final whistle. \n \nThere were {str(goals)} goals.")

#%%
## SIMULATING GOALS OVER A SEASON ##
#  We now simulate 380 matches of a football season and look at how well it predicts the
#  distribution of the number of goals. This is done in the code below: we loop over 380 matches,
#  store the number of goals for each match in array and then we make a histogram of the number of goals.
#

def simulate_match(n, p):
  # n - number of time units
  # p - probability per time unit of a goal
  # display_match == True then display simulation output for match.
  # Count the number of goals
  goals = 0
  for minute in range(n):
      # Generate a random number between 0 and 1.
      r = rnd.rand(1, 1)
      # Prints an X when there is a goal and a zero otherwise.
      if (r < p):
        # Goal - if the random number is less than the goal probability.
        goals = goals + 1

  return goals

# Number of matches
num_matches = 380

# Loop over all the matches and print the number of goals.
goals = np.zeros(num_matches)
for i in range(num_matches):
  goals[i] = simulate_match(match_minutes, prob_per_minute)
  #print("In match " + str(i+1) + " there were " + str(int(goals[i])) + " goals.")

# Create a histogram
fig, ax = plt.subplots(num=1)

histogram_range = np.arange(-0.5, 10.51, 1)
histogram_goals = np.histogram(goals, histogram_range)

ax.bar(histogram_goals[1][:-1] + 0.5, histogram_goals[0],
       color="white", edgecolor="black", linestyle="-", alpha=0.5)
ax.set_ylim(0, 100)

# formatting
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xticks(np.arange(0, 11, step=1))
ax.set_yticks(np.arange(0, 101, step=20))

# labels
ax.set_xlabel("Number of goals")
ax.set_ylabel("Number of matches")