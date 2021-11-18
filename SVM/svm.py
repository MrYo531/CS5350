import sys, os, random, math, decimal

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
    return [n * c for c in v]
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


# svm algorithm using stochastic sub-gradient descent
def svm(data, w, lr, t, c, a):
    for epoch in range(t):
        # update learning rate so weight is ensured to converge
        r = lr / (1 + (lr * epoch / a))
        #r = lr / (1 + epoch)

        # shuffle the data
        random.shuffle(data)

        # keep track of previous weight
        prev_w = w

        # loop through each data sample
        for x in data:
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
                w = scale(w, 1 - r) # don't update bias
                # using w[:-1] wasn't working   

        # Stop iterating when the length of the weight difference
        # (from the prev iteration) is less than the tolerance level
        w_diff = sub(prev_w, w)
        tolerance = 10e-3
        if norm(w_diff) < tolerance:
            print("converged at epoch:", epoch)
            return w

    return w


# reads the data and runs the perceptron algorithm to find the best weight vector
def main():
    # read the chosen method for the perceptron algorithm (standard, voted, or average)
    # perceptron_method = "standard" if len(sys.argv) == 1 else sys.argv[1]

    # define the file paths for the training and test data
    train_file = os.path.join("bank-note", "train.csv")
    test_file = os.path.join("bank-note", "test.csv")

    # read the data from the file into a list
    data = read_file(train_file)
    size = len(data[0])

    # setup init values
    w = [0] * size # folded b into w
    r = 0.1 # learning rate
    t = 100 # epoch
    c = 100/873 # hyperparameter
    a = 0.001 # for adjusting the learning rate

    # use the algorithm to calc the best weight vector
    learned_weight = svm(data, w, r, t, c, a)
    print("learned weight vector:", [round(num, 3) for num in learned_weight])

    # determine the average prediction error on the test data
    # errors = 0
    # test_data = read_file(test_file)
    # for x in test_data:
    #     # save the correct label because we will overwrite it
    #     label = x[-1]
    #     # and change any 0s to -1, that way it's easier to compare with
    #     label = -1.0 if label == 0 else label 
    #     # because we have b folded in w, the last value should be 1 so it can be multiplied through and have the bias be included in the final prediction value
    #     x[-1] = 1

    #     # find our prediction value by multiplying our weight vector with the data sample
    #     prediction = dot(learned_weight, x)

    #     # if misclassified, update our weight vector
    #     if label * prediction <= 0:
    #         errors += 1


    # error_percentage = errors / len(test_data)
    # print("[" + perceptron_method + "]", "average prediction error:", error_percentage)



if __name__ == "__main__":
    main()