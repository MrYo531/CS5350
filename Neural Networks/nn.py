import math, random, copy, decimal, os, sys

# reads the data from the given csv file
def read_file(CSV_file):
    data = []
    with open(CSV_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split(',')))
            data.append(values)
    f.close()
    return data

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def dot(x, y):
    return sum([x * y for x, y in zip(x, y)])

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

def forward_pass(x, weights, num_of_layers, nn_width, input_width, question):
    weight_value = 3
    
    # forward pass (calculate all the z values)
    z_values = []
    z_values.append(x)
    for layer in range(1, num_of_layers):
        z_values.append([1]) # the first element is the bias (1)
        for node_index in range(nn_width):
            prev_layer = layer - 1
            input = z_values[prev_layer]
            node_weight = []
            for i in range(input_width):
                index = (prev_layer * nn_width * input_width) + (nn_width * i) + node_index
                #print(weights[index])
                node_weight.append(weights[index][weight_value])
           
            z = sigmoid(dot(input, node_weight))
            z_values[layer].append(z)

    if question == "bp":   
        print("z_values: ", z_values)

    # y value/prediction (last calc)
    prev_layer = num_of_layers - 1
    input = z_values[prev_layer]
    node_weight = []
    for i in range(input_width):
        index = (prev_layer * nn_width * input_width) + i
        node_weight.append(weights[index][weight_value])
    
    y = dot(input, node_weight)
    if question == "bp": 
        print("y: ", y)

    return z_values, y

def back_propagation(x, weights, label, nn_width, input_width, question):
    num_of_layers = 3
    weight_value = 3

    # first forward pass (calculate all the z values)
    z_values, y = forward_pass(x, weights, num_of_layers, nn_width, input_width, question)

    partial_l_y = y - label
    top_layer = num_of_layers - 1 # (2)
    partial_derivatives = []
    for layer in reversed(range(1, num_of_layers + 1)):
        # top layer (3)
        if layer == num_of_layers: 
            for i in range(input_width):
                #index = (prev_layer * nn_width * input_width) + i
                #print(weights[index])
                input = z_values[top_layer]
                partial_y_w = input[i]
                partial_derivatives.append(partial_l_y * partial_y_w)

            continue
        
        # layer 2
        if layer == 2:
            for node_index in range(nn_width):
                next_layer = layer - 1
                for i in range(input_width):
                    index = (top_layer * nn_width * input_width) + node_index + 1
                    #print(weights[index])
                    partial_y_z = weights[index][weight_value]
                    node_to = z_values[top_layer][node_index + 1]
                    node_from = z_values[next_layer][i]
                    partial_z_w = node_to * (1 - node_to) * node_from
                    partial_derivatives.append(partial_l_y * partial_y_z * partial_z_w)
            
            continue
        

        # layer 1
        if layer == 1:
            for node_index in range(nn_width):
                next_layer = layer - 1
                for i in range(input_width):
                    sum = 0
                    for node_path_index in range(nn_width):
                        #print(node_index, i, node_path_index)
                        index = (top_layer * nn_width * input_width) + node_path_index + 1 
                        #print(weights[index])
                        partial_y_z = weights[index][weight_value]
                        #index = (layer * nn_width * input_width) + node_index + node_path_index + 1 
                        index = (layer * nn_width * input_width) + (nn_width * (i + 1)) + node_index + node_path_index
                        partial_z_z = weights[index][weight_value]
                        node_to = z_values[layer][node_index + 1]
                        node_from = z_values[next_layer][i]
                        partial_z_w = node_to * (1 - node_to) * node_from
                        sum += partial_l_y * partial_y_z * partial_z_z * partial_z_w
                    partial_derivatives.append(sum)

        if question == "bp": 
            print("partial derivatives: ", partial_derivatives)
        
        return partial_derivatives

# neural network algorithm using stochastic gradient descent
def nn(data, w, lr, t, a, nn_width, input_width, question):
    for epoch in range(t):
        # update learning rate so weight is ensured to converge
        r = lr / (1 + (lr * epoch / a))

        # shuffle the data
        random.shuffle(data)

        # keep track of previous weight
        prev_w = w

        # loop through each data sample
        for x in copy.deepcopy(data): # copy is important, otherwise we're changing the actual data (fixed bug)
            #w_0 = w[:-1] # without bias

            # save the y (label) value because we will overwrite it
            label = x[-1]

            # include bias
            x = [1] + x[:-1]

            #print(w)
            gradient = back_propagation(x, w, label, nn_width, input_width, question)
            print(gradient)

            return

            N = len(data) # data size 

     
            # and change any 0s to -1, that way it's easier to compare with
            y = -1.0 if y == 0 else y 
            # because we have b folded in w, the last value should be 1 so it can be multiplied through and have the bias be included in the final prediction value
            x[-1] = 1 

            # find our prediction value by multiplying our weight vector with the data sample
            wx = dot(w, x)

            # take the derivative of the svm objective at the current w to be the gradient J^t(w)
            # different sub-gradient values based on the prediction value
            # if y * wx <= 1:
            #     w = add(sub(w, scale((w_0 + [0]), r)), scale(x, r*c*N*y))
            # else:
            #     w[:-1] = scale(w[:-1], 1 - r) # don't update bias 
            w = sub(w, scale())

        # Stop iterating when the length of the weight difference vector
        # (from the prev iteration) is less than the tolerance level
        w_diff = sub(prev_w, w)
        tolerance = 10e-3
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

def gaussian_distrib(x):
    return ( 1 / (math.sqrt(2 * math.pi) ) ) * ( math.e ** ( (-1/2) * x**2 ) )

def main():
    # determine whether to call the primal or dual domain svm algorithm
    question = "bp" if len(sys.argv) <= 1 else sys.argv[1]

    # what each index represents
    node_layer = 0
    node_to = 1
    node_from = 2
    weight_value = 3

    # define all the weights
    # three layers
    num_of_layers = 3
    weights = []
    # layer 1
    weights.append( [ 1, 0, 1, -1] )
    weights.append( [ 1, 0, 2, 1 ] )
    weights.append( [ 1, 1, 1, -2 ] )
    weights.append( [ 1, 1, 2, 2 ] )
    weights.append( [ 1, 2, 1, -3 ] )
    weights.append( [ 1, 2, 2, 3 ] )
    # layer 2
    weights.append( [ 2, 0, 1, -1] )
    weights.append( [ 2, 0, 2, 1 ] )
    weights.append( [ 2, 1, 1, -2 ] )
    weights.append( [ 2, 1, 2, 2 ] )
    weights.append( [ 2, 2, 1, -3 ] )
    weights.append( [ 2, 2, 2, 3 ] )
    # layer 3 - output
    weights.append( [ 3, 0, 1, -1 ] )
    weights.append( [ 3, 1, 1, 2 ] )
    weights.append( [ 3, 2, 1, -1.5 ] )

    #print(weights)

    #print(z_values)

    # backpropagation - question a example
    
    if question == "bp":
        x = [1, 1, 1]
        label = 1
        nn_width = 2
        back_propagation(x, weights, label, nn_width, question)
    

    # define the file paths for the training and test data
    train_file = os.path.join("bank-note", "train.csv")
    test_file = os.path.join("bank-note", "test.csv")

    # read the data from the file into a list
    data = read_file(train_file)
    test_data = read_file(test_file)
    size = len(data[0])

    # setup init values
    w = [0] * size # folded b into w
    r = 0.0001 # learning rate
    t = 100 # epoch
    a = 0.0001 # for adjusting the learning rate
    
    
    x = [1, 1, 1]
    input_width = len(x)
    nn_width = 2
    num_of_layers = 3

    # init edge weights based on standard gaussian distribution
    weights = []
    num_of_weights = input_width * nn_width * (num_of_layers - 1)
    for i in range(num_of_weights):
        layer = int(i / (input_width * nn_width))
        node_from = int(i/2)  % input_width
        node_to = i % nn_width + 1
        weight_value = gaussian_distrib(random.uniform(0, 1))
        weights.append([layer, node_from, node_to, weight_value])
    for i in range(input_width):
        layer = 3
        node_from = i
        node_to = 1
        weight_value = gaussian_distrib(random.uniform(0, 1))
        weights.append([layer, node_from, node_to, weight_value])
    
    # use the algorithm to calc the best weight vector
    learned_weight = nn(data, weights, r, t, a, nn_width, input_width, question)

    #print("learned weight vector:", [round(num, 3) for num in learned_weight])

    #print_errors(learned_weight, data, test_data)



if __name__ == "__main__":
    main()