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
####  Training map

####  Features and output
Currently there are 32 features given to the network, 16 for the given unit and 16 for the given goal. These are the features in order  
[1] Unit Maximum Vitality - Continuous value (maximum health and shield combined )  
[2] Unit Movement mode - Binary categorical (Ground movement = 0, Flying = 1 )  
[3] Unit Weapon 1 Damage - Continuous value  
[4] Unit Weapon 1 Cooldown - Continuous value  
[5] Weapon 1 Can Attack Flying Units - Binary value  
[6] Weapon 1 Can Attack Ground Units - Binary value  
[7] Unit Health Regeneration - Continuous value  
[8] Unit Shield Regeneration - Continuous value  
[9] Unit Health Armor - Discrete value  
[10] Unit Shield Armor - Discrete value  
[11] Unit Maximum Health - Continuous value  
[12] Unit Maximum Shield - Continuous value  
[13] Unit Weapon 2 Damage - Continuous value  
[14] Unit Weapon 2 Cooldown - Continuous value  
[15] Weapon 2 Can Attack Air Units - Binary value  
[16] Weapon 2 Can Attack Flying Units - Binary value  
These features are given for both the given unit and the goal unit, combining into a 32 dimensional tensor.  
All of the values are different for each different type of unit.  
The values are normalized to a range of 0 and 1, based on this tensor  
[600,1,30,50,1,1,1,1,5,5,600,600,30,50,1,1,600,1,30,50,1,1,1,1,5,5,600,600,30,50,1,1]  
which is based on maximum values sampled from the units.  
  
The output value that is desired is this  
[0] Current vitality (pecentage) - Continuous value  
which is a number between 0 an 1 originally, however 1 was added such that the value was between 1 and 2 instead, as that provided much better results in the network.

#### Reinforcement Learning Test Map
Another map was set up where 2 random units were created next to each other, one for the AI and one set to attack the AIs unit. The state given to the AI is the prediction from the network based on the 2 units, the (x,y) distance between the units and both of their movement speeds. Every in-game-second the AI can then choose between 5 actions, move in one of the cardinal directions for one second, or attack for one second. After 10 seconds the current vitality percentage of the AI's unit is measured and the AI is rewarded based on the advantage between the measured vitality percentage and the prediction of the network, which is the predicted vitality percentage when only attacking. The units are then killed, two new are created and the cycle is repeated.
