# UVFA Algorithms
### Goal of project
The goal of this project is ultimately to create a functioning AI for playing StarCraft 2 (SC2), however to start with a more appropriate goal is set.

This goal is to create an effective way of predicting the outcome of a combat between two given units in SC2. To do this I want to use "Universal Value Function Approximation" (UVFA) as described by (LINK).
This is a complex algorithm and as such it was decided to work on the problem in iterations, each step focusing on a problem and relying on knowledge gained from previous iterations.

### Iteriantion 1 - Basic UVFA algorithm with OptSpace
The initial goal was to replicate examples given in the paper to ensure that the algorithm worked correctly. The example chosen for replication can be found on page (number), and was implemented using the second architecture described in the paper.
The architecture described relies on another algorithm, called OptSpace, to predict value function values for unencountered state/goal pairs, by training two neural networks to recreate this prediction process the UVFA paper hopes to create a third network that can accurately predict these value function values.

The programming environment that was chosen for this project was Python as it provides many useful machine learning libraries, furthermore a recent library has exposed the SC2 API to python thus making developing an AI for SC2 in Python much much easier.

####  Step 1 - Find OptSpace algorithm
Because the goal of this iteration was to replicate the given example in the UVFA paper
