import sys, os, random, math, decimal, copy
import numpy as np
from scipy.optimize import minimize

# reads the data from the given csv file
def read_file(CSV_file):
    data = []
    with open(CSV_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split(',')))
            data.append(values)
    f.close()
    return data

# dot product of two 1D vectors:
# multiplies the values of the two vectors together and returns the sum
def dot(a, b):
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

# scales a vector (v) by a number (n)
def scale(v, n):
    for i in range(len(v)):
        v[i] *= n
    return v

# adds one 1D vector to the other
def add(a, b):
    result = []
    for i in range(len(a)):
        result.append(a[i] + b[i])
    return result

# subtracts one 1D vector from the other
def sub(a, b):
        result = []
        for i in range(len(a)):
                result.append(a[i] - b[i])
        return result

# calculates the length of a vector
def norm(a):
        squared_sum = decimal.Decimal(0)
        for val in a:
                val = decimal.Decimal(val)
                squared_sum += val ** 2
        return math.sqrt(squared_sum)


# svm algorithm using stochastic sub-gradient descent (primal)
def svm(data, w, lr, t, c, a, schedule):
    for epoch in range(t):
        # update learning rate so weight is ensured to converge
        # schedule chooses which option
        if schedule == 0:
            r = lr / (1 + (lr * epoch / a))
        else:
            r = lr / (1 + epoch)

        # shuffle the data
        random.shuffle(data)

        # keep track of previous weight
        prev_w = w

        # loop through each data sample
        for x in copy.deepcopy(data): # copy is important, otherwise we're changing the actual data (fixed bug)
            w_0 = w[:-1] # without bias

            N = len(data) # data size 

            # save the y (label) value because we will overwrite it
            y = x[-1]
            # and change any 0s to -1, that way it's easier to compare with
            y = -1.0 if y == 0 else y 
            # because we have b folded in w, the last value should be 1 so it can be multiplied through and have the bias be included in the final prediction value
            x[-1] = 1 

            # find our prediction value by multiplying our weight vector with the data sample
            wx = dot(w, x)

            # take the derivative of the svm objective at the current w to be the gradient J^t(w)
            # different sub-gradient values based on the prediction value
            if y * wx <= 1:
                w = add(sub(w, scale((w_0 + [0]), r)), scale(x, r*c*N*y))
            else:
                w[:-1] = scale(w[:-1], 1 - r) # don't update bias 

        # Stop iterating when the length of the weight difference vector
        # (from the prev iteration) is less than the tolerance level
        w_diff = sub(prev_w, w)
        tolerance = 10e-3
        if norm(w_diff) < tolerance:
            print("converged at epoch:", epoch)
            return w

    return w

# quadratic convex equation that we want to minimize
def quad_conv_eq(alpha, *args):
    x = args[0]
    y = args[1]
    lr = args[2]

    if lr != -1:
        xx = x * x.T
    else:
        xx = np.exp(-np.sum(np.square(x - x)) / lr) # gaussian kernel

    yy = y * y.T
    aa = alpha * alpha.T

    #yyaaxx = yy * aa * xx # not working for some reason
    #yyaaxx = (yy * aa)[0, 0] * xx
    yyaaxx = alpha.T.dot((xx*yy)[0, 0] * alpha)

    return (1/2) * yyaaxx - np.sum(alpha)

# one of constraints for the quadratic convex equation
def constraint(alpha, *args):
    y = args[1]
    return np.sum(alpha * y)


# svm algorithm using stochastic sub-gradient descent (dual)
def svm_dual(data, w, c, lr):
    # set up x, y as numpy arrays from data
    size = len(data)
    x = np.matrix([data[i][:-1] for i in range(size)])
    y = np.matrix([data[i][-1] for i in range(size)]).T
    # replace 0s with -1
    y = np.where(y == 0, -1, y)

    # define the parameters for the minimize optimizer
    #initial_guess = np.zeros(size) # this just gives us 0
    #initial_guess = np.full((size), 1) # this gives us 640 support vectors..
    initial_guess = np.random.rand(size) # this is inconsistent...
    
    args = (x, y, lr)
    method = 'SLSQP'
    bounds = [(0, c)] * size
    constraints = [{'type': 'eq', 'fun' : constraint, 'args' : args}]

    # find the optimal alpha by minimizing the quadratic convex equation
    res = minimize(quad_conv_eq, initial_guess, args, method, bounds = bounds, constraints = constraints)
    best_alpha = res.x
    
    #print(best_alpha)
    
    # get number of support vectors
    support_vectors = np.where(best_alpha > 0)[0]
    print("num of support vectors:", len(support_vectors))

    # calculate weight
    w = np.sum((best_alpha * y)[0, 0] * x, axis=0)
    w = np.matrix.tolist(w)[0]
    b = np.sum((best_alpha * y)[0, 0] * (x * x.T))
    b = np.matrix.tolist(b)
    w.append(b)

    return w

def print_errors(learned_weight, data, test_data):
    # ------------------------
    # TRAINING AND TEST ERRORS
    # ------------------------

    # determine the average prediction error on the training data
    errors = 0
    for x in copy.deepcopy(data):
        # save the correct label because we will overwrite it
        label = x[-1]
        # and change any 0s to -1, that way it's easier to compare with
        label = -1.0 if label == 0 else label 
        # because we have b folded in w, the last value should be 1 so it can be multiplied through and have the bias be included in the final prediction value
        x[-1] = 1

        # find our prediction value by multiplying our weight vector with the data sample
        prediction = dot(learned_weight, x)

        # if misclassified, update our weight vector
        if label * prediction <= 0:
            errors += 1

    error_percentage = errors / len(data)
    print("training error:", error_percentage)


    # determine the average prediction error on the test data
    errors = 0
    for x in copy.deepcopy(test_data):
        # save the correct label because we will overwrite it
        label = x[-1]
        # and change any 0s to -1, that way it's easier to compare with
        label = -1.0 if label == 0 else label 
        # because we have b folded in w, the last value should be 1 so it can be multiplied through and have the bias be included in the final prediction value
        x[-1] = 1

        # find our prediction value by multiplying our weight vector with the data sample
        prediction = dot(learned_weight, x)

        # if misclassified, update our weight vector
        if label * prediction <= 0:
            errors += 1


    error_percentage = errors / len(test_data)
    print("test error:", error_percentage)

# reads the data and runs the perceptron algorithm to find the best weight vector
def main():
    # read and set the c value
    c = 100/873 if len(sys.argv) == 1 else float(sys.argv[1])
    #print("c:", c)

    # read and set the schedule option
    schedule = 0 if len(sys.argv) <= 2 else float(sys.argv[2])

    # determine whether to call the primal or dual domain svm algorithm
    domain = "primal" if len(sys.argv) <= 3 else sys.argv[3]

    # determine whether use either the linear or gaussian kernel
    kernel = "linear" if len(sys.argv) <= 4 else sys.argv[4]


    # define the file paths for the training and test data
    train_file = os.path.join("bank-note", "train.csv")
    test_file = os.path.join("bank-note", "test.csv")
    #train_file = os.path.join("SVM", "bank-note", "train.csv") # for debugging
    #test_file = os.path.join("SVM", "bank-note", "test.csv")

    # read the data from the file into a list
    data = read_file(train_file)
    test_data = read_file(test_file)
    size = len(data[0])

    # setup init values
    w = [0] * size # folded b into w
    r = 0.0001 # learning rate
    t = 100 # epoch
    #c = 100/873 # hyperparameter
    a = 0.0001 # for adjusting the learning rate
    gaussian_lr = [0.1, 0.5, 1, 5, 100]

    if domain == "primal":
        # use the algorithm to calc the best weight vector
        learned_weight = svm(data, w, r, t, c, a, schedule)
    else:
        if kernel == "linear":
            learned_weight = svm_dual(data, w, c, -1)
        else:
            for lr in gaussian_lr:
                print("gaussian learning rate:", lr)
                learned_weight = svm_dual(data, w, c, lr)

                print("learned weight vector:", [round(num, 3) for num in learned_weight])
                print_errors(learned_weight, data, test_data)
            return


    print("learned weight vector:", [round(num, 3) for num in learned_weight])

    print_errors(learned_weight, data, test_data)


if __name__ == "__main__":
    main()