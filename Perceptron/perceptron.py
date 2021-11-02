import sys, os, random

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


# perceptron algorithm (standard version)
def perceptron(data, w, r, t):
    for _ in range(t):
        # shuffle the data
        random.shuffle(data)

        # loop through each data sample
        for x in data:
             # save the y (label) value because we will overwrite it
            y = x[-1]
            # and change any 0s to -1, that way it's easier to compare with
            y = -1.0 if y == 0 else y 
            # because we have b folded in w, the last value should be 1 so it can be multiplied through and have the bias be included in the final prediction value
            x[-1] = 1 
            
            # find our prediction value by multiplying our weight vector with the data sample
            wx = dot(w, x)

            # if misclassified, update our weight vector
            if y * wx <= 0:
                w = add(w, scale(x, r * y))
        #print(w)
    return w

# perceptron algorithm (voted version)
def perceptron_voted(data, w, r, t):
    m = 0 # number of weights
    Cm = 0 # number of correct predictions for each weight
    weights = [] # keeps track of multiple weights
    weights.append((w, Cm)) # add the first weight so the algorithm can use it

    for _ in range(t):
        # loop through each data sample
        for x in data:
             # save the y (label) value because we will overwrite it
            y = x[-1]
            # and change any 0s to -1, that way it's easier to compare with
            y = -1.0 if y == 0 else y 
            # because we have b folded in w, the last value should be 1 so it can be multiplied through and have the bias be included in the final prediction value
            x[-1] = 1 
            
            # find our prediction value by multiplying our weight vector with the data sample
            wx = dot(weights[m][0], x)

            # if misclassified, update our weight vector and how many correct values
            if y * wx <= 0:
                # the count for the first weight needs to be corrected
                if len(weights) == 1:
                    weights[0] = (w, Cm)

                weights.append((add(weights[m][0], scale(x, r * y)), Cm))
                m += 1
                Cm = 1
            else:
                Cm += 1
        #print(w)
    return weights

# perceptron algorithm (average version)
def perceptron_avg(data, w, r, t):
    # 'a' will represent the sum of all the different weights
    a = w
    for _ in range(t):
        # loop through each data sample
        for x in data:
             # save the y (label) value because we will overwrite it
            y = x[-1]
            # and change any 0s to -1, that way it's easier to compare with
            y = -1.0 if y == 0 else y 
            # because we have b folded in w, the last value should be 1 so it can be multiplied through and have the bias be included in the final prediction value
            x[-1] = 1 
            
            # find our prediction value by multiplying our weight vector with the data sample
            wx = dot(w, x)

            # if misclassified, update our weight vector
            if y * wx <= 0:
                w = add(w, scale(x, r * y))

            # add the current weight for every iteration
            a = add(a, w)
        #print(w)
    return a



# reads the data and runs the perceptron algorithm to find the best weight vector
def main():
    # read the chosen method for the perceptron algorithm (standard, voted, or average)
    perceptron_method = "standard" if len(sys.argv) == 1 else sys.argv[1]

    # define the file paths for the training and test data
    train_file = os.path.join("bank-note", "train.csv")
    test_file = os.path.join("bank-note", "test.csv")

    # read the data from the file into a list
    data = read_file(train_file)
    size = len(data[0])

    # setup init values
    w = [0] * size # folded b into w
    r = 0.1 # learning rate
    t = 10 # epoch

    if perceptron_method == "standard":
    
        # use the algorithm to calc the best weight vector
        learned_weight = perceptron(data, w, r, t)
        print("learned weight vector:", [round(num, 3) for num in learned_weight])

        # determine the average prediction error on the test data
        errors = 0
        test_data = read_file(test_file)
        for x in test_data:
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
        print("[" + perceptron_method + "]", "average prediction error:", error_percentage)

    elif perceptron_method == "voted":
        
        # use the algorithm to calc the weight vectors and their correct predictions
        learned_weights = perceptron_voted(data, w, r, t)
        
        for i, weight in enumerate(learned_weights):
            print("[" + str(i) + "]", "learned weight vector:", [round(num, 3) for num in weight[0]], "| Cm:", weight[1])

        # determine the average prediction error on the test data
        errors = 0
        test_data = read_file(test_file)
        for x in test_data:
            # save the correct label because we will overwrite it
            label = x[-1]
            # and change any 0s to -1, that way it's easier to compare with
            label = -1.0 if label == 0 else label 
            # because we have b folded in w, the last value should be 1 so it can be multiplied through and have the bias be included in the final prediction value
            x[-1] = 1

            # find our prediction value by multiplying our weight vector with the data sample 
            # and by the Cm value FOR EACH vector. then sum
            prediction_sum = 0
            for weight in learned_weights:
                prediction = dot(weight[0], x)
                sign = 1 if prediction > 0 else -1
                prediction_sum += weight[1] * sign

            # if misclassified, update our weight vector
            if label * prediction_sum <= 0:
                errors += 1

        error_percentage = errors / len(test_data)
        print("[" + perceptron_method + "]", "average prediction error:", error_percentage)

    elif perceptron_method == "average":
        
        # use the algorithm to calc the best weight vector
        learned_weight = perceptron_avg(data, w, r, t)
        print("learned weight vector:", [round(num, 3) for num in learned_weight])

        # determine the average prediction error on the test data
        errors = 0
        test_data = read_file(test_file)
        for x in test_data:
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
        print("[" + perceptron_method + "]", "average prediction error:", error_percentage)

    else:
        print("Invalid command")





if __name__ == "__main__":
    main()