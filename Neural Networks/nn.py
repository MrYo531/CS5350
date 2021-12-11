import math

def nn():
    NotImplemented

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def dot(x, y):
    return sum([x * y for x, y in zip(x, y)])

def main():

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

    # forward pass (calculate all the z values)
    nn_width = 2
    z_values = []
    x = [1, 1, 1]
    input_width = len(x)
    z_values.append(x)
    for layer in range(1, num_of_layers):
        z_values.append([1]) # the first element is the bias (1)
        for node_index in range(nn_width):
            prev_layer = layer - 1
            input = z_values[prev_layer]
            node_weight = []
            for l in range(input_width):
                index = (prev_layer * nn_width * input_width) + (nn_width * l) + node_index
                node_weight.append(weights[index][weight_value])
           
            z = sigmoid(dot(input, node_weight))
            z_values[layer].append(z)
        
    print(z_values)

    # y value/prediction (last calc)
    prev_layer = num_of_layers - 1
    input = z_values[prev_layer]
    node_weight = []
    for l in range(input_width):
        index = (prev_layer * nn_width * input_width) + l
        node_weight.append(weights[index][weight_value])
    
    y = dot(input, node_weight)
    print(y)

    # backpropagation



if __name__ == "__main__":
    main()