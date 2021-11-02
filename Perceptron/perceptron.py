import os

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


# perceptron algorithm
def perceptron(data, w, r, t):
    for i in range(1): # FIXME
        # maybe keep track of errors, so we can break early
        for x in data:
            label = x[-1] # save the label value because we will overwrite it
            label = -1.0 if label == 0 else label # and change any 0s to -1, that way it's easier to compare with
            x[-1] = 1 # because we have b folded in w, the last attrib should be 1 so it will be multiplied through and have the bias included in the final prediction value
            
            prediction = dot(w, x)
            print(prediction)
            print(label)

            if label * prediction <= 0:
                print("not matching")
            else:
                print("matching")

            print()

    return w


# reads the data and runs the perceptron algorithm to find the best weight vector
def main():

    train_file = os.path.join("bank-note", "train.csv")
    test_file = os.path.join("bank-note", "test.csv")

    data = read_file(train_file)
    dsize = len(data[0])

    w = [-1] * dsize # folded b into w
    r = 0.1 # learning rate
    t = 10 # epoch

    best_weight = perceptron(data, w, r, t)
    print(best_weight)

if __name__ == "__main__":
    main()