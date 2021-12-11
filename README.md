# CS5350
This is a machine learning library developed by Kidus Yohannes for CS5350/6350 in University of Utah

For the decision tree command "dt.py", simply run the command with two arguments. The first argument is the type of purity (error measurement): "entropy", "me", "gi". The second arguement is the depth of the tree to be created.

For example, the followng command will create a tree using majority error and a depth of 2: 
"python3 ./dt.py me 2"

Update:

Passing "ada" for the first argument will run the adaboost algorithm. And the second arguement is the forest size.

By default boost is used as the ensemble method. Passing "bag" as the third parameter will using bagging, "random" will using random forests.

For example, the followng command will use adaboost to create a forest of size 5. Using the bagging algorithm to choose random data points:
"python3 ./dt.py ada 5 bag"

In the Linear Regression folder is the gradient descent and stochastic gradient descent algorithm. Simply run the script to see the best weight and cost function value for the test data.

For example:
"python3 '.\stochastic gradient descent.py'"

Update:

For the Perceptron algorithm, simply run the command with one argument, which specifies with version to run. Either "standard", "voted", or "average".

For example, the followng command will run the perceptron algorithm using the average version: 
"python3 .\perceptron.py average"

Update:

For the SVM algorithm, the script takes 4 parameters:
- The first is the hyperparameter C (float)
- The second is the schedule version, which determines which equation to dynamically change the learning rate (0 or 1)
- The third is the whether to run the "primal" or "dual" version of SVM
- The fourth is whether to use a "linear" or "gaussian" kernal

Update:

For the NN algorithm, the script takes 1 parameters:
- Type 'bp' to see the results for the backpropagation algorithm tested on the example data given by the assignment
- Type 'sgd' to see the results for the stochastic gradient descent algorithm that uses backpropagation to calcualte the weight gradients
- Type 'sgd-0' to see a variation where the initial weights are all 0, instead of based on the gaussian distribution.

