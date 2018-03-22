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




### Iteration 3 - StarCraft 2 implementation

#### Theory
##### Compounding interest of units
As described before there are several different units and unit types in SC2, and their relative strengths might not always be immediately predictable if they are grouped together. This concept is what will be called "compound interest" in this paper, a term taken from economics where one in theory could get an ever increasing amount of money with the same amount of interest if the money gained from interest was put back into the account which the interest was calculated from. The idea in SC2 is best explained with a practical example: imagine that you have one marine, a ranged unit, versus one zergling, a mellee unit, and they start x units apart. After t amount of time the zergling travels the x units and reaches the marine, however in this time the marine has dealt 50% of the zerglings health in damage. After another t amount of time the zergling is dead as the marine dealth the remaining 50% of its health as damage, however the zergling also damaged the marine for 33% of his health, due to the marines higher armor and health in comparison to the zergling. Now imagine the same scenario, but with two marines and two zerglings instead, still x units apart. This time after t amount of time has passed, one zergling is already dead, due to the two marines being able to deal a combined 100% of its health as damage in t amount of time, but the other one is at full health and has arrived at the marines. After another t amoutn of time one of the marines has taken the 33% of its health as damage, like before, and the other zergling has died too. Now, by doubling the marines and the zerglings in number, the damage dealt by the marines was doubled too however the damage taken remained the same for one marine but the other one left unscathed. Had an AI, or a person, been asked to predict the outcome of the second battle based on the results of the first, it might have said that each marine should take 33% damage before both zerglings were dead, 
