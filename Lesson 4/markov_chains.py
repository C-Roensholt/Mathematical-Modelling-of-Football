#%%
# Lesson 4: Markov Chains
# https://soccermatics.readthedocs.io/en/latest/gallery/lesson4/plot_MarkovChain.html

import numpy as np

# Pass matrix
A = np.matrix([[0.25, 0.20, 0.1], [0.1, 0.25, 0.2],[0.1, 0.1, 0.25]])
# Goal vector
g = np.transpose(np.matrix([0.05, 0.15, 0.05]))

## Linear Algebra Method ##
# Solve (I-A)xT = g
xT1 = np.linalg.solve(np.identity(3) - A,g)
print('Expected Threat\nCentral, Box, Wing')
print(np.transpose(xT1))

## Iterative Method ##
# Iterate xT’ = A xT + g to update through each move of the ball
xT2 = np.zeros((3,1))
for t in range(10):
   #print(np.matmul(A,xT2) + g)
   xT2 = np.matmul(A,xT2) + g
print('Expected Threat')
print('Central, Box, Wing')
print(np.transpose(xT2))


## Monte Carlo Simulation ##
# Simulate "num_sims" possessions, starting from each of the three areas/state
num_sims = 10
xT3 = np.zeros(3)

description = {0: 'Central', 1: 'Wing', 2: 'Box' }

for i in range(3):
    num_goals = 0

    print('---------------')
    print('Start from ' + description[i] )
    print('---------------')

    for n in range(num_sims):

        ballinplay=True
        #Initial state is i
        s = i
        describe_possession=''

        while ballinplay:
            r=np.random.rand()

            # Make commentary text
            describe_possession = describe_possession + ' - ' + description[s]


            #Cumulative sum of in play probabilities
            c_sum=np.cumsum(A[s,:])
            new_s = np.sum(r>c_sum)
            if new_s>2:
                #Ball is either goal or out of play
                ballinplay=False
                if r < g[s] + c_sum[0,2]:
                    #Its a goal!
                    num_goals = num_goals + 1
                    describe_possession = describe_possession + ' - Goal!'
                else:
                    describe_possession = describe_possession + ' - Out of play'
            s = new_s

        print(describe_possession)

    xT3[i] = num_goals/num_sims


print('\n\n---------------')
print('Expected Threat')
print('Central, Box, Wing')
print(xT3)