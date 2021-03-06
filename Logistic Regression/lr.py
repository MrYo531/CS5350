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

def avg(data):
    sum = [0] * (len(data[0]) - 1)
    N = len(data)
    for i in range(N):
        x = data[i][:-1]
        sum = add(sum, x)
    return scale(sum, 1 / N)

def variance(data):
    sum = [0] * (len(data[0]) - 1)
    N = len(data)
    #for i in range(N):
    x = data[0][:-1]
    dif = sub(x, avg(data))
    # squared
    for j in range(len(dif)):
        dif[j] *= dif[j]
    sum = add(sum, x)
    return scale(sum, 1 / 1)

def map_est(x, y, w, m, variance):
    x[-1] = 1 
    gradient = [0] * len(x)

    frac = (math.e ** (-y * dot(w, x))) / (1 + math.e ** (-y * dot(w, x)) )
    s = scale(x, -y)
    gradient = scale(s, frac)
    #gradient = scale(gradient, m)
    gradient = add(gradient, scale(w, 2 / sum(variance)))
    return gradient

def ml_est(x, y, w, m):
    x[-1] = 1 
    gradient = [1] * len(x)
    ml = math.log(1 + math.e ** (-y * dot(w, x)))
    #frac = (math.e ** (-y * dot(w, x))) / (1 + math.e ** (-y * dot(w, x)) )
    #s = scale(x, -y)
    #gradient = scale(s, frac)
    #gradient = scale(gradient, m)
    gradient = scale(gradient, ml)
    return gradient

# logistic_regression algorithm using ML and MAP estimation
def logistic_regression(data, w, lr, t, c, a, estimation):
    for epoch in range(t):
        # update learning rate so weight is ensured to converge
        r = lr / (1 + (lr * epoch / a))

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

            # calculate gradient - map or ml
            gradient = map_est(x, y, w, N, variance(data)) if estimation == "map" else ml_est(x, y, w, N)
            #print(gradient)

            # find our prediction value by multiplying our weight vector with the data sample
            #wx = dot(w, x)
            wx = dot(gradient, x)

            # take the derivative of the svm objective at the current w to be the gradient J^t(w)
            # different sub-gradient values based on the prediction value
            if y * wx <= 1:
                w = add(sub(w, scale((w_0 + [0]), r)), scale(x, r*c*N*y))
            else:
                w[:-1] = scale(w[:-1], 1 - r) # don't update bias 

        # Stop iterating when the length of the weight difference vector
        # (from the prev iteration) is less than the tolerance level
        w_diff = sub(prev_w, w)
        tolerance = 10e-1
        if norm(w_diff) < tolerance:
            print("converged at epoch:", epoch)
            return w

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
    # read and set the estimation option
    estimation = "map" if len(sys.argv) <= 1 else sys.argv[1]

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
    c = 100/873 # hyperparameter
    a = 0.0001 # for adjusting the learning rate

    learned_weight = logistic_regression(data, w, r, t, c, a, estimation)

    print("learned weight vector:", [round(num, 3) for num in learned_weight])

    print_errors(learned_weight, data, test_data)


if __name__ == "__main__":
    main()